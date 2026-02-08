
#!/usr/bin/env python3
"""
Japanese Learning App - Terminal Interface
A beautiful curses-based interface for learning Japanese with spaced repetition.
Requires OpenAI API for intelligent exercise generation and evaluation.
"""

import curses
import sys
import os
import json
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime
import traceback
import base64
from pathlib import Path

# Check for debug mode
DEBUG = os.environ.get("DEBUG", "0") == "1"
MAX_COMPLETION_TOKENS = 32768

# Check for OpenAI API key before proceeding
if "OPENAI_API_KEY" not in os.environ:
    print("Error: OPENAI_API_KEY environment variable is required")
    print("Please set it with: export OPENAI_API_KEY='your-api-key-here'")
    sys.exit(1)

try:
    import openai
except ImportError:
    print("Error: OpenAI library is required")
    print("Please install it with: pip install openai")
    sys.exit(1)

# Initialize OpenAI client
from openai import OpenAI
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Add the current directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_learn_japanese import db, exercises
from llm_learn_japanese.db import CardAspect, Progress


class OpenAIModel:
    """Wrapper for OpenAI API to match expected interface."""
    def __init__(self, client: OpenAI, model_name: str = "gpt-5-mini"):
        self.client = client
        self.model_name = model_name

    def prompt(self, prompt_text: str, system: str = "") -> Any:
        """Send prompt to OpenAI and return response."""
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt_text})

        # Log the API call details
        if DEBUG:
            print(f"🤖 OpenAI API Call Details:")
            print(f"   Model: {self.model_name}")
            print(f"   System prompt length: {len(system) if system else 0} characters")
            print(f"   User prompt length: {len(prompt_text)} characters")
            print(f"   Temperature: 1.0")
            print(f"   Max tokens: 2500")
            if system:
                print(f"   System prompt preview: {system[:100]}...")
            print(f"   User prompt preview: {prompt_text[:200]}...")

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,  # type: ignore
                temperature=1.0,
                max_completion_tokens=MAX_COMPLETION_TOKENS,
            )

            content = response.choices[0].message.content or ""

            # Log the response details
            if DEBUG:
                print(f"✅ OpenAI API Response:")
                print(f"   Response length: {len(content)} characters")
                print(f"   Usage: {response.usage}")
                if hasattr(response, 'model'):
                    print(f"   Model used: {response.model}")
                print(f"   Response preview: {content[:300]}...")

            # Return object with text() method to match expected interface
            class Response:
                def __init__(self, content: str) -> None:
                    self.content = content
                def text(self) -> str:
                    return self.content

            return Response(content)

        except Exception as e:
            if DEBUG:
                print(f"❌ OpenAI API call failed: {str(e)}")
                print(f"   Error type: {type(e).__name__}")
            raise


class JapaneseApp:
    def __init__(self, stdscr: Any) -> None:
        self.stdscr = stdscr
        self.height, self.width = stdscr.getmaxyx()

        # Initialize AI model - REQUIRED
        self.ai_model = OpenAIModel(client, model_name="gpt-5-mini")

        # Initialize colors
        curses.start_color()
        curses.use_default_colors()

        # Define color pairs
        curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLUE)    # Header
        curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_WHITE)   # Selected item
        curses.init_pair(3, curses.COLOR_GREEN, -1)                  # Success
        curses.init_pair(4, curses.COLOR_RED, -1)                    # Error
        curses.init_pair(5, curses.COLOR_YELLOW, -1)                 # Warning
        curses.init_pair(6, curses.COLOR_CYAN, -1)                   # Info
        curses.init_pair(7, curses.COLOR_MAGENTA, -1)                # Accent

        # Configure cursor and input
        curses.curs_set(0)  # Hide cursor
        self.stdscr.keypad(True)
        # Don't set timeout - wait for user input

        self.current_menu = "main"
        self.menu_index = 0
        self.message = ""
        self.message_color = 0
        self.username = "default_user"  # Default username for progress tracking

    def _draw_base_screen(self) -> None:
        """Draws the common elements of the screen."""
        self.stdscr.clear()
        self.draw_header()
        self.draw_footer()
        self.draw_message()

    def draw_header(self) -> None:
        """Draw the application header."""
        header_text = "🇯🇵 Japanese Learning App 🇯🇵 [AI-Powered]"
        self.stdscr.attron(curses.color_pair(1) | curses.A_BOLD)
        self.stdscr.addstr(0, (self.width - len(header_text)) // 2, header_text)
        self.stdscr.attroff(curses.color_pair(1) | curses.A_BOLD)

        # Draw separator line
        self.stdscr.addstr(1, 0, "─" * self.width)

    def draw_footer(self) -> None:
        """Draw navigation help footer."""
        footer_text = "↑↓: Navigate | Enter: Select | Q: Quit | B: Back"
        footer_y = self.height - 1
        self.stdscr.attron(curses.color_pair(6))
        self.stdscr.addstr(footer_y, (self.width - len(footer_text)) // 2, footer_text)
        self.stdscr.attroff(curses.color_pair(6))

    def show_message(self, message: str, color: int = 0) -> None:
        """Display a status message."""
        self.message = message
        self.message_color = color

    def draw_message(self) -> None:
        """Draw the status message if there is one."""
        if self.message:
            y = self.height - 3
            if self.message_color > 0:
                self.stdscr.attron(curses.color_pair(self.message_color))
            self.stdscr.addstr(y, 2, f"Status: {self.message}")
            if self.message_color > 0:
                self.stdscr.attroff(curses.color_pair(self.message_color))

    def draw_menu(self, title: str, options: List[str], selected_index: int) -> None:
        """Draw a menu with the given options."""
        start_y = 4

        # Draw title with bounds checking
        try:
            if start_y < self.height - 1 and len(title) < self.width - 4:
                self.stdscr.attron(curses.color_pair(7) | curses.A_BOLD)
                self.stdscr.addstr(start_y, 2, title[:self.width - 4])
                self.stdscr.attroff(curses.color_pair(7) | curses.A_BOLD)
        except curses.error:
            pass  # Skip if drawing fails

        # Draw options with bounds checking
        for i, option in enumerate(options):
            y = start_y + 2 + i
            if y >= self.height - 1:  # Stop if we're at the bottom
                break

            try:
                # Remove problematic characters and truncate if necessary
                safe_option = option.replace('🇯🇵', 'JP').replace('📚', '').replace('➕', '+').replace('📊', '').replace('👥', '').replace('✍️', '').replace('📝', '').replace('🤖', 'AI').replace('🗄️', '').replace('🚪', '')
                safe_option = safe_option.strip()[:self.width - 8]  # Leave room for prefix

                if i == selected_index:
                    self.stdscr.attron(curses.color_pair(2) | curses.A_BOLD)
                    self.stdscr.addstr(y, 4, f"> {safe_option}")
                    self.stdscr.attroff(curses.color_pair(2) | curses.A_BOLD)
                else:
                    self.stdscr.addstr(y, 4, f"  {safe_option}")
            except curses.error:
                pass  # Skip problematic characters

    def get_user_input(self, prompt: str, multiline: bool = False) -> Optional[str]:
        """Get text input from the user with improved support for Japanese input."""
        curses.curs_set(1)  # Show cursor
        self._draw_base_screen()

        # Draw prompt
        y = self.height // 2 - 3
        self.stdscr.attron(curses.color_pair(6) | curses.A_BOLD)
        prompt_lines = self.wrap_text(prompt, self.width - 4)
        for i, line in enumerate(prompt_lines):
            self.stdscr.addstr(y + i, 2, line)
        self.stdscr.attroff(curses.color_pair(6) | curses.A_BOLD)

        # Draw input box
        input_y = y + len(prompt_lines) + 1
        input_x = 2
        max_input_length = self.width - 10

        self.stdscr.addstr(input_y, input_x, "Input: ")
        self.stdscr.addstr(input_y + 2, 2, "Enter: Submit | Esc: Cancel | Paste supported")

        if multiline:
            self.stdscr.addstr(input_y + 3, 2, "Ctrl+D: End multiline input")

        self.stdscr.refresh()

        input_buffer: List[str] = []
        input_str = ""
        lines: Optional[List[str]] = [] if multiline else None

        while True:
            # Clear and redraw input area
            self.stdscr.addstr(input_y, input_x + 7, " " * max_input_length)
            display_str = input_str[-max_input_length:] if len(input_str) > max_input_length else input_str
            self.stdscr.addstr(input_y, input_x + 7, display_str)
            self.stdscr.move(input_y, input_x + 7 + len(display_str))
            self.stdscr.refresh()

            try:
                ch = self.stdscr.get_wch()  # Proper Unicode-aware input
            except curses.error:
                continue

            if ch == "\n" or ch == "\r":  # Enter key
                if multiline and lines is not None:
                    lines.append("".join(input_buffer))
                    input_buffer = []
                    input_str = ""
                    input_y += 1
                    if input_y >= self.height - 4:
                        break
                else:
                    curses.curs_set(0)
                    return "".join(input_buffer).strip()
            elif ch == "\x04" and multiline:  # Ctrl+D for multiline
                if lines is not None:
                    lines.append("".join(input_buffer))
                    curses.curs_set(0)
                    return "\n".join(lines).strip()
            elif ch == "\x1b":  # ESC key
                curses.curs_set(0)
                return None
            elif ch in ("\b", "\x7f"):  # Backspace
                if input_buffer:
                    input_buffer.pop()
                    input_str = "".join(input_buffer)
            elif ch == "\x17":  # Ctrl+W - delete word
                if input_buffer:
                    temp_str = "".join(input_buffer).rstrip()
                    last_space = temp_str.rfind(" ")
                    if last_space == -1:
                        input_buffer = []
                    else:
                        input_buffer = list(temp_str[:last_space + 1])
                    input_str = "".join(input_buffer)
            elif isinstance(ch, str):  # Any valid Unicode character
                input_buffer.append(ch)
                input_str = "".join(input_buffer)

        return None  # Fallback return

    def show_study_session(self) -> None:
        """Display an enhanced study session interface with AI-powered exercises."""
        cards_studied = 0
        correct_count = 0

        while True:
            aspect = db.get_next_card()
            if not aspect:
                self._draw_base_screen()
                self.stdscr.attron(curses.color_pair(3) | curses.A_BOLD)
                self.stdscr.addstr(4, 2, "🎉 No cards due for review! All caught up!")
                self.stdscr.attroff(curses.color_pair(3) | curses.A_BOLD)

                if cards_studied > 0:
                    accuracy = (correct_count / cards_studied * 100) if cards_studied > 0 else 0
                    self.stdscr.addstr(6, 2, f"Session Summary:")
                    self.stdscr.addstr(7, 4, f"Cards studied: {cards_studied}")
                    self.stdscr.addstr(8, 4, f"Correct: {correct_count}")
                    self.stdscr.addstr(9, 4, f"Accuracy: {accuracy:.1f}%")

                self.stdscr.addstr(11, 2, "Press any key to return to menu...")
                self.stdscr.refresh()
                self.stdscr.getch()
                return

            self._draw_base_screen()

            # Generate AI-powered exercise
            self.stdscr.addstr(3, 2, "🤖 Generating AI exercise...")
            self.stdscr.refresh()

            try:
                if DEBUG:
                    print(f"🎯 Generating exercise for aspect:")
                    print(f"   Type: {aspect.parent_type} - {aspect.aspect_type}")
                    print(f"   Aspect ID: {aspect.id}")
                    print(f"   Success count: {aspect.success_count}")
                question = exercises.generate_exercise(aspect, model=self.ai_model)
                if DEBUG:
                    print(f"✅ Exercise generated successfully: {question[:100]}...")
            except Exception as e:
                if DEBUG:
                    print(f"❌ AI exercise generation failed: {str(e)}")
                    print(f"   Error type: {type(e).__name__}")
                    import traceback
                    print(f"   Traceback: {traceback.format_exc()}")
                self.show_message(f"❌ AI generation failed: {str(e)}", 4)
                self.stdscr.addstr(5, 2, "Press any key to skip this card...")
                self.stdscr.refresh()
                self.stdscr.getch()
                continue

            self._draw_base_screen()

            # Display session info
            self.stdscr.attron(curses.color_pair(1))
            self.stdscr.addstr(2, 2, f" Study Session - Cards: {cards_studied + 1} | Correct: {correct_count} ")
            self.stdscr.attroff(curses.color_pair(1))

            # Display question
            self.stdscr.attron(curses.color_pair(6) | curses.A_BOLD)
            self.stdscr.addstr(4, 2, f"Aspect Type: {aspect.aspect_type.replace('_', ' ').title()}")
            self.stdscr.attroff(curses.color_pair(6) | curses.A_BOLD)

            # Wrap question text
            question_lines = self.wrap_text(question, self.width - 4)
            for i, line in enumerate(question_lines):
                self.stdscr.addstr(6 + i, 2, line)

            # Show options
            y = 6 + len(question_lines) + 1
            self.stdscr.attron(curses.color_pair(6))
            self.stdscr.addstr(y, 2, "Options: [Enter] Answer | [C] Chat | [S] Skip | [Q] Quit Session")
            self.stdscr.attroff(curses.color_pair(6))
            self.stdscr.refresh()

            # Get user choice
            key = self.stdscr.getch()

            if key == ord('c') or key == ord('C'):
                # Multi-turn chat feature
                self.handle_learning_chat(aspect, question)
                continue

            elif key == ord('q') or key == ord('Q'):
                if cards_studied > 0:
                    self._draw_base_screen()
                    accuracy = (correct_count / cards_studied * 100) if cards_studied > 0 else 0
                    self.stdscr.addstr(4, 2, f"Session ended early")
                    self.stdscr.addstr(6, 2, f"Cards studied: {cards_studied}")
                    self.stdscr.addstr(7, 2, f"Correct: {correct_count}")
                    self.stdscr.addstr(8, 2, f"Accuracy: {accuracy:.1f}%")
                    self.stdscr.addstr(10, 2, "Press any key to continue...")
                    self.stdscr.refresh()
                    self.stdscr.getch()
                return

            elif key == ord('s') or key == ord('S'):
                # Skip this card
                cards_studied += 1
                continue

            elif key == ord('\n') or key == ord('\r'):
                # Get user answer
                user_answer = self.get_user_input("Enter your answer:")
                if user_answer is None:
                    continue

                # AI-powered evaluation
                self._draw_base_screen()
                self.stdscr.addstr(4, 2, "🤖 AI is evaluating your answer...")
                self.stdscr.refresh()

                try:
                    if DEBUG:
                        print(f"🧠 Evaluating answer with AI:")
                        print(f"   User answer: {user_answer}")
                        print(f"   Aspect: {aspect.parent_type} - {aspect.aspect_type}")
                    score, feedback = db.evaluate_answer_with_ai(aspect, user_answer, model=self.ai_model)
                    if DEBUG:
                        print(f"✅ AI evaluation completed:")
                        print(f"   Score: {score}/5")
                        print(f"   Feedback: {feedback[:100]}...")
                except Exception as e:
                    if DEBUG:
                        print(f"❌ AI evaluation failed: {str(e)}")
                        print(f"   Error type: {type(e).__name__}")
                        import traceback
                        print(f"   Traceback: {traceback.format_exc()}")
                    self.show_message(f"❌ AI evaluation failed: {str(e)}", 4)
                    score = 3  # Default moderate score
                    feedback = "AI evaluation unavailable, manual review recommended."

                # Update progress tracking
                cards_studied += 1
                is_correct = score >= 3
                if is_correct:
                    correct_count += 1

                db.update_progress(self.username, is_correct)
                db.review_card(aspect.id, score)

                # Show feedback
                self._draw_base_screen()

                # Color-code the result
                if score >= 4:
                    color = 3  # Green
                    symbol = "✅"
                elif score >= 3:
                    color = 5  # Yellow
                    symbol = "⚠️"
                else:
                    color = 4  # Red
                    symbol = "❌"

                self.stdscr.attron(curses.color_pair(color) | curses.A_BOLD)
                self.stdscr.addstr(4, 2, f"{symbol} Score: {score}/5")
                self.stdscr.attroff(curses.color_pair(color) | curses.A_BOLD)

                self.stdscr.addstr(6, 2, f"Your answer: {user_answer}")

                # Wrap feedback text
                feedback_lines = self.wrap_text(f"AI Feedback: {feedback}", self.width - 4)
                for i, line in enumerate(feedback_lines):
                    self.stdscr.addstr(8 + i, 2, line)

                y = 8 + len(feedback_lines) + 1
                if score >= 3:
                    self.stdscr.attron(curses.color_pair(3))
                    self.stdscr.addstr(y, 2, "📅 Card rescheduled for later review")
                    self.stdscr.attroff(curses.color_pair(3))
                else:
                    self.stdscr.attron(curses.color_pair(5))
                    self.stdscr.addstr(y, 2, "🔄 Card will be reviewed again soon")
                    self.stdscr.attroff(curses.color_pair(5))

                self.stdscr.addstr(y + 2, 2, "Press any key to continue to next card...")
                self.stdscr.refresh()
                self.stdscr.getch()

    def handle_learning_chat(self, aspect: Any, question: str) -> None:
        """Handle multi-turn chat conversation about the current learning topic."""
        chat_history: list[tuple[str, str]] = []

        # Initialize with the current question context
        system_prompt = f"""You are a helpful Japanese language tutor. The student is currently learning about:

Topic: {aspect.aspect_type.replace('_', ' ').title()}
Current Question: {question}

Your role is to:
1. Answer questions about this topic in a clear, educational manner
2. Provide additional context, examples, or explanations when helpful
3. Encourage the student and provide supportive feedback
4. Stay focused on Japanese language learning

Keep responses concise but informative (2-3 sentences max). Use simple English explanations for complex Japanese concepts."""

        while True:
            self._draw_base_screen()

            # Show chat header
            self.stdscr.attron(curses.color_pair(1))
            self.stdscr.addstr(2, 2, f" Learning Chat - {aspect.aspect_type.replace('_', ' ').title()} ")
            self.stdscr.attroff(curses.color_pair(1))

            # Show current topic
            topic_lines = self.wrap_text(f"Topic: {question}", self.width - 4)
            y = 4
            for i, line in enumerate(topic_lines):
                if y + i < self.height - 8:
                    self.stdscr.attron(curses.color_pair(6))
                    self.stdscr.addstr(y + i, 2, line[:self.width - 4])
                    self.stdscr.attroff(curses.color_pair(6))
            y += len(topic_lines) + 1

            # Show recent chat history (last 4 exchanges, 8 messages)
            recent_history = chat_history[-8:] if len(chat_history) > 8 else chat_history

            for i, (role, message) in enumerate(recent_history):
                if y >= self.height - 8:  # Leave space for input area
                    break

                if role == "user":
                    self.stdscr.attron(curses.color_pair(7))
                    prefix = "You: "
                else:
                    self.stdscr.attron(curses.color_pair(3))
                    prefix = "Tutor: "

                # Wrap message text more carefully
                wrapped_lines = self.wrap_text(f"{prefix}{message}", self.width - 4)
                for j, line in enumerate(wrapped_lines):
                    if y + j < self.height - 8:
                        try:
                            self.stdscr.addstr(y + j, 2, line[:self.width - 4])
                        except curses.error:
                            pass  # Skip if can't draw
                self.stdscr.attroff(curses.color_pair(7) if role == "user" else curses.color_pair(3))
                y += len(wrapped_lines) + 1

            # Show input prompt and options
            input_y = self.height - 4
            try:
                self.stdscr.attron(curses.color_pair(6))
                self.stdscr.addstr(input_y, 2, "Type your question (or 'exit' to return to study session):")
                self.stdscr.attroff(curses.color_pair(6))
                self.stdscr.refresh()
            except curses.error:
                pass

            # Get user input with better wrapping
            user_message = self.get_multiline_input("Your question:")
            if not user_message or user_message.lower().strip() in ['exit', 'quit', 'back']:
                break

            # Add user message to history
            chat_history.append(("user", user_message))

            # Show "AI thinking" indicator
            self._draw_base_screen()
            self.stdscr.attron(curses.color_pair(6))
            self.stdscr.addstr(4, 2, "🤖 Tutor is thinking...")
            self.stdscr.attroff(curses.color_pair(6))
            self.stdscr.refresh()

            try:
                # Get AI response
                if DEBUG:
                    print(f"💬 Learning chat - User: {user_message}")

                # Build conversation context
                conversation = []
                for role, msg in chat_history[-6:]:  # Last 6 messages for context
                    conversation.append(f"{'Student' if role == 'user' else 'Tutor'}: {msg}")

                conversation_context = "\n".join(conversation)

                ai_response = self.ai_model.prompt(
                    f"Previous conversation:\n{conversation_context}\n\nPlease respond as a helpful Japanese tutor to the student's latest question.",
                    system=system_prompt
                )

                tutor_response = ai_response.text().strip()
                if DEBUG:
                    print(f"💬 Learning chat - Tutor: {tutor_response[:100]}...")

                # Add AI response to history
                chat_history.append(("assistant", tutor_response))

                # Show the AI response immediately
                self._draw_base_screen()
                self.stdscr.attron(curses.color_pair(1))
                self.stdscr.addstr(2, 2, f" Tutor Response ")
                self.stdscr.attroff(curses.color_pair(1))

                # Display the response
                response_lines = self.wrap_text(f"Tutor: {tutor_response}", self.width - 4)
                y = 4
                for i, line in enumerate(response_lines):
                    if y + i < self.height - 4:
                        try:
                            self.stdscr.attron(curses.color_pair(3))
                            self.stdscr.addstr(y + i, 2, line[:self.width - 4])
                            self.stdscr.attroff(curses.color_pair(3))
                        except curses.error:
                            pass

                # Show continue prompt
                try:
                    self.stdscr.attron(curses.color_pair(6))
                    self.stdscr.addstr(self.height - 2, 2, "Press any key to continue the conversation...")
                    self.stdscr.attroff(curses.color_pair(6))
                    self.stdscr.refresh()
                    self.stdscr.getch()  # Wait for user to acknowledge
                except curses.error:
                    pass

            except Exception as e:
                if DEBUG:
                    print(f"❌ Learning chat failed: {str(e)}")
                tutor_response = "I'm sorry, I'm having trouble right now. Could you try asking again?"
                chat_history.append(("assistant", tutor_response))

                # Show error message
                self._draw_base_screen()
                self.stdscr.attron(curses.color_pair(4))
                self.stdscr.addstr(4, 2, f"Error: {tutor_response}")
                self.stdscr.attroff(curses.color_pair(4))
                self.stdscr.addstr(6, 2, "Press any key to continue...")
                self.stdscr.refresh()
                self.stdscr.getch()

        # Return to study session
        return

    def get_multiline_input(self, prompt: str) -> Optional[str]:
        """Get text input with better wrapping and multiline support."""
        curses.curs_set(1)  # Show cursor
        self._draw_base_screen()

        # Draw prompt
        y = self.height // 2 - 4
        self.stdscr.attron(curses.color_pair(6) | curses.A_BOLD)
        prompt_lines = self.wrap_text(prompt, self.width - 4)
        for i, line in enumerate(prompt_lines):
            if y + i < self.height - 6:
                try:
                    self.stdscr.addstr(y + i, 2, line[:self.width - 4])
                except curses.error:
                    pass
        self.stdscr.attroff(curses.color_pair(6) | curses.A_BOLD)

        # Draw input area
        input_start_y = y + len(prompt_lines) + 1
        input_x = 2
        max_input_width = self.width - 6

        try:
            self.stdscr.addstr(input_start_y, input_x, "Input: ")
            self.stdscr.addstr(input_start_y + 1, 2, "Enter: Submit | Esc: Cancel | Text will wrap automatically")
            self.stdscr.refresh()
        except curses.error:
            pass

        input_buffer: List[str] = []
        display_lines: List[str] = []
        cursor_pos = 0

        while True:
            # Update display lines based on current buffer
            full_text = "".join(input_buffer)
            display_lines = self.wrap_text(full_text, max_input_width)
            if not display_lines:
                display_lines = [""]

            # Clear input area and redraw
            for i in range(min(4, self.height - input_start_y - 3)):  # Show up to 4 lines
                try:
                    self.stdscr.addstr(input_start_y + 2 + i, input_x + 7, " " * max_input_width)
                    if i < len(display_lines):
                        display_text = display_lines[i][:max_input_width]
                        self.stdscr.addstr(input_start_y + 2 + i, input_x + 7, display_text)
                except curses.error:
                    pass

            # Position cursor (simplified)
            try:
                cursor_line = min(len(display_lines) - 1, 3)
                cursor_col = len(display_lines[cursor_line]) if cursor_line < len(display_lines) else 0
                self.stdscr.move(input_start_y + 2 + cursor_line, input_x + 7 + cursor_col)
                self.stdscr.refresh()
            except curses.error:
                pass

            try:
                ch = self.stdscr.get_wch()
            except curses.error:
                continue

            if ch == "\n" or ch == "\r":  # Enter key
                curses.curs_set(0)
                return "".join(input_buffer).strip()
            elif ch == "\x1b":  # ESC key
                curses.curs_set(0)
                return None
            elif ch in ("\b", "\x7f"):  # Backspace
                if input_buffer:
                    input_buffer.pop()
            elif ch == "\x17":  # Ctrl+W - delete word
                if input_buffer:
                    temp_str = "".join(input_buffer).rstrip()
                    last_space = temp_str.rfind(" ")
                    if last_space == -1:
                        input_buffer = []
                    else:
                        input_buffer = list(temp_str[:last_space + 1])
            elif isinstance(ch, str) and len(ch) == 1:  # Valid character
                input_buffer.append(ch)

        return None

    def wrap_text(self, text: str, max_width: int) -> List[str]:
        """Wrap text to fit within max_width, with improved support for Japanese text."""
        if not text or max_width <= 0:
            return [text] if text else []

        # First try word-based wrapping for English/mixed content
        words = text.split()
        lines = []
        current_line = ""

        for word in words:
            # Check if this word alone exceeds max_width
            if len(word) > max_width:
                # If we have content in current_line, add it first
                if current_line:
                    lines.append(current_line)
                    current_line = ""

                # Break the long word into chunks
                for i in range(0, len(word), max_width):
                    chunk = word[i:i + max_width]
                    lines.append(chunk)
            else:
                # Normal word processing
                test_line = current_line + " " + word if current_line else word
                if len(test_line) <= max_width:
                    current_line = test_line
                else:
                    if current_line:
                        lines.append(current_line)
                    current_line = word

        if current_line:
            lines.append(current_line)

        # If we only got one long line and it contains no spaces (likely Japanese),
        # fall back to character-based wrapping
        if len(lines) == 1 and len(lines[0]) > max_width and ' ' not in text:
            lines = []
            current_line = ""

            for char in text:
                if len(current_line + char) <= max_width:
                    current_line += char
                else:
                    if current_line:
                        lines.append(current_line)
                    current_line = char

            if current_line:
                lines.append(current_line)

        return lines if lines else [text]

    def show_add_vocabulary(self) -> None:
        """Interface for adding vocabulary."""
        word = self.get_user_input("Enter Japanese word/phrase:")
        if word:
            is_new = db.add_card(word)
            if is_new:
                self.show_message(f"✅ Added vocabulary: {word}", 3)
            else:
                self.show_message(f"⚠️ Vocabulary already exists: {word}", 5)

    def show_add_kanji(self) -> None:
        """Interface for adding kanji."""
        character = self.get_user_input("Enter kanji character:")
        if not character:
            return

        onyomi = self.get_user_input("Enter onyomi reading (optional):") or ""
        kunyomi = self.get_user_input("Enter kunyomi reading (optional):") or ""
        meanings = self.get_user_input("Enter meanings (optional):") or ""

        is_new = db.add_kanji(character, onyomi, kunyomi, meanings)
        if is_new:
            self.show_message(f"✅ Added kanji: {character}", 3)
        else:
            self.show_message(f"⚠️ Kanji already exists: {character}", 5)

    def show_add_grammar(self) -> None:
        """Interface for adding grammar points."""
        point = self.get_user_input("Enter grammar point:")
        if not point:
            return

        explanation = self.get_user_input("Enter explanation (optional):") or ""
        example = self.get_user_input("Enter example sentence (optional):") or ""

        is_new = db.add_grammar(point, explanation, example)
        if is_new:
            self.show_message(f"✅ Added grammar: {point}", 3)
        else:
            self.show_message(f"⚠️ Grammar already exists: {point}", 5)

    def show_add_idiom(self) -> None:
        """Interface for adding idioms."""
        idiom = self.get_user_input("Enter idiom:")
        if not idiom:
            return

        meaning = self.get_user_input("Enter meaning (optional):") or ""
        example = self.get_user_input("Enter example sentence (optional):") or ""

        is_new = db.add_idiom(idiom, meaning, example)
        if is_new:
            self.show_message(f"✅ Added idiom: {idiom}", 3)
        else:
            self.show_message(f"⚠️ Idiom already exists: {idiom}", 5)

    def show_progress_stats(self) -> None:
        """Display user progress statistics."""
        self._draw_base_screen()

        # Get progress for current user
        progress = db.get_progress(self.username)

        y = 4
        self.stdscr.attron(curses.color_pair(7) | curses.A_BOLD)
        self.stdscr.addstr(y, 2, f"📊 Progress Statistics for {self.username}")
        self.stdscr.attroff(curses.color_pair(7) | curses.A_BOLD)

        y += 2
        if progress:
            accuracy = (progress.correct_answers / progress.total_reviews * 100) if progress.total_reviews > 0 else 0

            self.stdscr.addstr(y, 4, f"Total Reviews: {progress.total_reviews}")
            y += 1
            self.stdscr.addstr(y, 4, f"Correct Answers: {progress.correct_answers}")
            y += 1

            # Color code accuracy
            if accuracy >= 80:
                color = 3  # Green
            elif accuracy >= 60:
                color = 5  # Yellow
            else:
                color = 4  # Red

            self.stdscr.attron(curses.color_pair(color))
            self.stdscr.addstr(y, 4, f"Accuracy: {accuracy:.1f}%")
            self.stdscr.attroff(curses.color_pair(color))
            y += 1

            self.stdscr.addstr(y, 4, f"Last Updated: {progress.last_updated.strftime('%Y-%m-%d %H:%M')}")
        else:
            self.stdscr.addstr(y, 4, "No progress data yet. Start studying to track your progress!")

        y += 2
        self.stdscr.attron(curses.color_pair(6))
        self.stdscr.addstr(y, 2, "Press any key to continue...")
        self.stdscr.attroff(curses.color_pair(6))
        self.stdscr.refresh()
        self.stdscr.getch()

    def process_image_file(self) -> None:
        """Process an image file to extract Japanese content using AI vision."""
        filepath = self.get_user_input("Enter image file path:")
        if not filepath:
            return

        # Check if file exists
        if not os.path.exists(filepath):
            self.show_message(f"❌ File not found: {filepath}", 4)
            return

        self._draw_base_screen()
        self.stdscr.addstr(4, 2, "🤖 Processing image with AI vision...")
        self.stdscr.addstr(6, 2, "This feature uses GPT-5 to:")
        self.stdscr.addstr(7, 4, "1. Extract text from image using OCR")
        self.stdscr.addstr(8, 4, "2. Identify Japanese content")
        self.stdscr.addstr(9, 4, "3. Categorize into vocabulary, kanji, grammar, etc.")
        self.stdscr.addstr(10, 4, "4. Add items to database")
        self.stdscr.refresh()

        try:
            # Read and encode image
            if DEBUG:
                print(f"📁 Reading image file: {filepath}")
            with open(filepath, 'rb') as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')

            # Create vision model prompt
            from llm_learn_japanese import structured
            prompt = structured.PROMPT_INSTRUCTIONS

            # Use OpenAI vision model
            messages = [
                {"role": "system", "content": "You are a Japanese learning assistant."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            }
                        }
                    ]
                }
            ]

            # Log the API request details
            if DEBUG:
                print(f"🤖 Making vision API call to: gpt-5-2025-08-07")
                print(f"📝 Prompt length: {len(prompt)} characters")
                print(f"🖼️  Image data length: {len(image_data)} characters")
                print(f"📋 System prompt: You are a Japanese learning assistant.")
                print(f"📋 User prompt preview: {prompt[:200]}...")

            response = client.chat.completions.create(
                model="gpt-5-2025-08-07",
                messages=messages,  # type: ignore
                max_completion_tokens=MAX_COMPLETION_TOKENS,
                temperature=1.0,  # Lower temperature for more consistent extraction
            )

            content = response.choices[0].message.content or "{}"

            # Log the raw response
            if DEBUG:
                print(f"📤 Raw API response length: {len(content)} characters")
                print(f"📤 Raw API response preview: {content[:500]}...")
                if content != "{}":
                    print(f"📤 Raw API response (full):\n{content}")

            try:
                structured_content = json.loads(content)
                if DEBUG:
                    print(f"✅ JSON parsing successful")
                    print(f"📊 Parsed structure keys: {list(structured_content.keys()) if isinstance(structured_content, dict) else 'Not a dict'}")
            except json.JSONDecodeError as e:
                if DEBUG:
                    print(f"❌ JSON parsing failed: {e}")
                    print(f"🔍 Attempting to extract JSON from response...")
                # Try to extract JSON from response if it's wrapped in other text
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    try:
                        structured_content = json.loads(json_match.group())
                        if DEBUG:
                            print(f"✅ Extracted JSON successfully")
                    except json.JSONDecodeError:
                        if DEBUG:
                            print(f"❌ Even extracted JSON failed to parse")
                        structured_content = {"vocabulary": [], "kanji": [], "grammar": [], "phrases": [], "idioms": []}
                else:
                    if DEBUG:
                        print(f"❌ No JSON found in response")
                    structured_content = {"vocabulary": [], "kanji": [], "grammar": [], "phrases": [], "idioms": []}

            # Log the final structured content before processing
            if DEBUG:
                print(f"📊 Final structured content:")
                for key, items in structured_content.items():
                    print(f"   {key}: {len(items)} items")
                    if items:
                        print(f"      First item example: {items[0] if items else 'None'}")

            # Process all structured content types with duplicate tracking
            vocab_new = vocab_dup = kanji_new = kanji_dup = grammar_new = grammar_dup = phrase_new = phrase_dup = idiom_new = idiom_dup = 0

            # Vocabulary
            vocabulary_items = structured_content.get("vocabulary", [])
            if DEBUG:
                print(f"🔍 Processing {len(vocabulary_items)} vocabulary items...")
            for i, item in enumerate(vocabulary_items):
                if DEBUG:
                    print(f"   📝 Vocab item {i+1}/{len(vocabulary_items)}: {item}")
                word = item.get("word", "") if isinstance(item, dict) else ""
                if word and word.strip():
                    if DEBUG:
                        print(f"   🔄 Checking if '{word}' already exists in database...")
                    if db.add_card(word):
                        vocab_new += 1
                        if DEBUG:
                            print(f"   ✅ Added new vocabulary: {word}")
                    else:
                        vocab_dup += 1
                        if DEBUG:
                            print(f"   ⚠️  Vocabulary already exists: {word}")
                else:
                    if DEBUG:
                        print(f"   ❌ Invalid vocabulary item: {item}")
            if DEBUG:
                print(f"✅ Vocabulary processing complete: {vocab_new} new, {vocab_dup} duplicates")

            # Kanji
            kanji_items = structured_content.get("kanji", [])
            if DEBUG:
                print(f"🔍 Processing {len(kanji_items)} kanji items...")
            for i, item in enumerate(kanji_items):
                if DEBUG:
                    print(f"   🈷️  Kanji item {i+1}/{len(kanji_items)}: {item}")
                character = item.get("character", "") if isinstance(item, dict) else ""
                if character and character.strip():
                    if DEBUG:
                        print(f"   🔄 Checking if kanji '{character}' already exists in database...")
                    if db.add_kanji(
                        character=character,
                        onyomi=item.get("onyomi", ""),
                        kunyomi=item.get("kunyomi", ""),
                        meanings=item.get("meanings", "")
                    ):
                        kanji_new += 1
                        if DEBUG:
                            print(f"   ✅ Added new kanji: {character}")
                    else:
                        kanji_dup += 1
                        if DEBUG:
                            print(f"   ⚠️  Kanji already exists: {character}")
                else:
                    if DEBUG:
                        print(f"   ❌ Invalid kanji item: {item}")
            if DEBUG:
                print(f"✅ Kanji processing complete: {kanji_new} new, {kanji_dup} duplicates")

            # Grammar
            grammar_items = structured_content.get("grammar", [])
            if DEBUG:
                print(f"🔍 Processing {len(grammar_items)} grammar items...")
            for i, item in enumerate(grammar_items):
                if DEBUG:
                    print(f"   📝 Grammar item {i+1}/{len(grammar_items)}: {item}")
                point = item.get("point", "") if isinstance(item, dict) else ""
                if point and point.strip():
                    if DEBUG:
                        print(f"   🔄 Checking if grammar '{point}' already exists in database...")
                    if db.add_grammar(
                        point=point,
                        explanation=item.get("explanation", ""),
                        example=item.get("example", "")
                    ):
                        grammar_new += 1
                        if DEBUG:
                            print(f"   ✅ Added new grammar: {point}")
                    else:
                        grammar_dup += 1
                        if DEBUG:
                            print(f"   ⚠️  Grammar already exists: {point}")
                else:
                    if DEBUG:
                        print(f"   ❌ Invalid grammar item: {item}")
            if DEBUG:
                print(f"✅ Grammar processing complete: {grammar_new} new, {grammar_dup} duplicates")

            # Phrases
            phrase_items = structured_content.get("phrases", [])
            if DEBUG:
                print(f"🔍 Processing {len(phrase_items)} phrase items...")
            for i, item in enumerate(phrase_items):
                if DEBUG:
                    print(f"   💬 Phrase item {i+1}/{len(phrase_items)}: {item}")
                phrase = item.get("phrase", "") if isinstance(item, dict) else ""
                if phrase and phrase.strip():
                    if DEBUG:
                        print(f"   🔄 Checking if phrase '{phrase}' already exists in database...")
                    if db.add_phrase(
                        phrase=phrase,
                        meaning=item.get("meaning", "")
                    ):
                        phrase_new += 1
                        if DEBUG:
                            print(f"   ✅ Added new phrase: {phrase}")
                    else:
                        phrase_dup += 1
                        if DEBUG:
                            print(f"   ⚠️  Phrase already exists: {phrase}")
                else:
                    if DEBUG:
                        print(f"   ❌ Invalid phrase item: {item}")
            if DEBUG:
                print(f"✅ Phrase processing complete: {phrase_new} new, {phrase_dup} duplicates")

            # Idioms
            idiom_items = structured_content.get("idioms", [])
            if DEBUG:
                print(f"🔍 Processing {len(idiom_items)} idiom items...")
            for i, item in enumerate(idiom_items):
                if DEBUG:
                    print(f"   🎭 Idiom item {i+1}/{len(idiom_items)}: {item}")
                idiom = item.get("idiom", "") if isinstance(item, dict) else ""
                if idiom and idiom.strip():
                    if DEBUG:
                        print(f"   🔄 Checking if idiom '{idiom}' already exists in database...")
                    if db.add_idiom(
                        idiom=idiom,
                        meaning=item.get("meaning", ""),
                        example=item.get("example", "")
                    ):
                        idiom_new += 1
                        if DEBUG:
                            print(f"   ✅ Added new idiom: {idiom}")
                    else:
                        idiom_dup += 1
                        if DEBUG:
                            print(f"   ⚠️  Idiom already exists: {idiom}")
                else:
                    if DEBUG:
                        print(f"   ❌ Invalid idiom item: {item}")
            if DEBUG:
                print(f"✅ Idiom processing complete: {idiom_new} new, {idiom_dup} duplicates")

            total_new = vocab_new + kanji_new + grammar_new + phrase_new + idiom_new
            total_dup = vocab_dup + kanji_dup + grammar_dup + phrase_dup + idiom_dup

            if DEBUG:
                print(f"\n🎉 PROCESSING COMPLETE!")
                print(f"📊 Final Results:")
                print(f"   ✅ Added: {vocab_new} vocabulary, {kanji_new} kanji, {grammar_new} grammar, {phrase_new} phrases, {idiom_new} idioms")
                print(f"   ⚠️  Skipped duplicates: {vocab_dup} vocabulary, {kanji_dup} kanji, {grammar_dup} grammar, {phrase_dup} phrases, {idiom_dup} idioms")
                print(f"   📈 Total new items: {total_new}")
                print(f"   🔄 Total duplicates: {total_dup}")

            # Display results
            self._draw_base_screen()
            self.stdscr.addstr(4, 2, "✅ Image processed successfully!")
            self.stdscr.addstr(6, 2, f"Added: {vocab_new} vocabulary, {kanji_new} kanji, {grammar_new} grammar, {phrase_new} phrases, {idiom_new} idioms")
            if total_dup > 0:
                self.stdscr.addstr(7, 2, f"Skipped duplicates: {vocab_dup} vocabulary, {kanji_dup} kanji, {grammar_dup} grammar, {phrase_dup} phrases, {idiom_dup} idioms")

        except Exception as e:
            self._draw_base_screen()
            self.stdscr.addstr(4, 2, f"❌ Image processing failed: {str(e)}")

        self.stdscr.addstr(10, 2, "Press any key to continue...")
        self.stdscr.refresh()
        self.stdscr.getch()

    def process_multiple_images(self) -> None:
        """Process multiple image files using brace expansion patterns."""
        pattern = self.get_user_input("Enter file pattern (e.g., /path/file_{001..010}.jpg):")
        if not pattern:
            return

        self._draw_base_screen()
        self.stdscr.addstr(4, 2, "🔍 Expanding file pattern...")
        self.stdscr.refresh()

        try:
            # Try to import braceexpand
            try:
                from braceexpand import braceexpand
            except ImportError:
                self._draw_base_screen()
                self.stdscr.addstr(4, 2, "❌ braceexpand module not installed")
                self.stdscr.addstr(6, 2, "Please install it with: pip install braceexpand")
                self.stdscr.addstr(8, 2, "Press any key to continue...")
                self.stdscr.refresh()
                self.stdscr.getch()
                return

            # Expand the pattern to get all file paths
            file_paths = list(braceexpand(pattern))
            if DEBUG:
                print(f"🔍 Pattern expanded to {len(file_paths)} files")
                print(f"📁 Files: {file_paths[:5]}{'...' if len(file_paths) > 5 else ''}")

            # Filter to only existing files
            existing_files = [f for f in file_paths if os.path.exists(f)]
            missing_files = [f for f in file_paths if not os.path.exists(f)]

            if missing_files and DEBUG:
                print(f"⚠️  {len(missing_files)} files not found: {missing_files[:3]}{'...' if len(missing_files) > 3 else ''}")

            if not existing_files:
                self._draw_base_screen()
                self.stdscr.addstr(4, 2, f"❌ No files found matching pattern: {pattern}")
                self.stdscr.addstr(6, 2, "Press any key to continue...")
                self.stdscr.refresh()
                self.stdscr.getch()
                return

            self._draw_base_screen()
            self.stdscr.addstr(4, 2, f"🎯 Found {len(existing_files)} images to process")
            self.stdscr.addstr(6, 2, "Processing will begin in 3 seconds...")
            self.stdscr.addstr(8, 2, "Press any key to start immediately, or wait...")
            self.stdscr.refresh()

            # Give user a chance to cancel
            import time
            self.stdscr.timeout(3000)  # 3 second timeout
            key = self.stdscr.getch()
            self.stdscr.timeout(-1)  # Reset to blocking

            # Cumulative statistics
            total_vocab_new = total_kanji_new = total_grammar_new = total_phrase_new = total_idiom_new = 0
            total_vocab_dup = total_kanji_dup = total_grammar_dup = total_phrase_dup = total_idiom_dup = 0
            processed_count = 0
            failed_count = 0

            if DEBUG:
                print(f"\n🚀 Starting batch processing of {len(existing_files)} images...")

            for i, filepath in enumerate(existing_files, 1):
                if DEBUG:
                    print(f"\n📸 Processing image {i}/{len(existing_files)}: {os.path.basename(filepath)}")

                self._draw_base_screen()
                self.stdscr.addstr(4, 2, f"🤖 Processing image {i}/{len(existing_files)}")
                self.stdscr.addstr(6, 2, f"File: {os.path.basename(filepath)}")
                self.stdscr.addstr(8, 2, f"Progress: {(i-1)/len(existing_files)*100:.1f}% complete")
                self.stdscr.refresh()

                try:
                    # Process single image
                    vocab_new, kanji_new, grammar_new, phrase_new, idiom_new, vocab_dup, kanji_dup, grammar_dup, phrase_dup, idiom_dup = self._process_single_image(filepath)

                    # Add to cumulative totals
                    total_vocab_new += vocab_new
                    total_kanji_new += kanji_new
                    total_grammar_new += grammar_new
                    total_phrase_new += phrase_new
                    total_idiom_new += idiom_new
                    total_vocab_dup += vocab_dup
                    total_kanji_dup += kanji_dup
                    total_grammar_dup += grammar_dup
                    total_phrase_dup += phrase_dup
                    total_idiom_dup += idiom_dup
                    processed_count += 1

                    if DEBUG:
                        print(f"✅ Image {i} complete: +{vocab_new}v, +{kanji_new}k, +{grammar_new}g, +{phrase_new}p, +{idiom_new}i")

                except Exception as e:
                    if DEBUG:
                        print(f"❌ Failed to process {filepath}: {str(e)}")
                    failed_count += 1
                    continue

            # Show final results
            if DEBUG:
                print(f"\n🎉 BATCH PROCESSING COMPLETE!")
                print(f"📊 Final Statistics:")
                print(f"   ✅ Successfully processed: {processed_count}/{len(existing_files)} images")
                print(f"   ❌ Failed: {failed_count}/{len(existing_files)} images")
                print(f"   📈 Total new items: {total_vocab_new} vocab, {total_kanji_new} kanji, {total_grammar_new} grammar, {total_phrase_new} phrases, {total_idiom_new} idioms")
                print(f"   🔄 Total duplicates: {total_vocab_dup} vocab, {total_kanji_dup} kanji, {total_grammar_dup} grammar, {total_phrase_dup} phrases, {total_idiom_dup} idioms")

            self._draw_base_screen()
            self.stdscr.addstr(4, 2, "🎉 Batch processing complete!")
            self.stdscr.addstr(6, 2, f"Processed: {processed_count}/{len(existing_files)} images")
            self.stdscr.addstr(7, 2, f"Added: {total_vocab_new} vocab, {total_kanji_new} kanji, {total_grammar_new} grammar")
            self.stdscr.addstr(8, 2, f"       {total_phrase_new} phrases, {total_idiom_new} idioms")
            self.stdscr.addstr(9, 2, f"Duplicates: {total_vocab_dup + total_kanji_dup + total_grammar_dup + total_phrase_dup + total_idiom_dup} total")

        except Exception as e:
            self._draw_base_screen()
            self.stdscr.addstr(4, 2, f"❌ Batch processing failed: {str(e)}")

        self.stdscr.addstr(11, 2, "Press any key to continue...")
        self.stdscr.refresh()
        self.stdscr.getch()

    def _process_single_image(self, filepath: str) -> tuple:
        """Process a single image and return statistics."""
        # Read and encode image
        if DEBUG:
            print(f"📁 Reading image file: {filepath}")
        with open(filepath, 'rb') as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')

        # Create vision model prompt
        from llm_learn_japanese import structured
        prompt = structured.PROMPT_INSTRUCTIONS

        # Use OpenAI vision model
        messages = [
            {"role": "system", "content": "You are a Japanese learning assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}"
                        }
                    }
                ]
            }
        ]

        # Log the API request details
        if DEBUG:
            print(f"🤖 Making vision API call to: gpt-5-2025-08-07")
            print(f"📝 Prompt length: {len(prompt)} characters")
            print(f"🖼️  Image data length: {len(image_data)} characters")

        response = client.chat.completions.create(
            model="gpt-5-2025-08-07",
            messages=messages,  # type: ignore
            max_completion_tokens=MAX_COMPLETION_TOKENS,
            temperature=1.0,
        )

        content = response.choices[0].message.content or "{}"

        # Log the raw response
        if DEBUG:
            print(f"📤 Raw API response length: {len(content)} characters")

        try:
            structured_content = json.loads(content)
            if DEBUG:
                print(f"✅ JSON parsing successful")
        except json.JSONDecodeError as e:
            if DEBUG:
                print(f"❌ JSON parsing failed: {e}")
            # Try to extract JSON from response if it's wrapped in other text
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                try:
                    structured_content = json.loads(json_match.group())
                    if DEBUG:
                        print(f"✅ Extracted JSON successfully")
                except json.JSONDecodeError:
                    if DEBUG:
                        print(f"❌ Even extracted JSON failed to parse")
                    structured_content = {"vocabulary": [], "kanji": [], "grammar": [], "phrases": [], "idioms": []}
            else:
                if DEBUG:
                    print(f"❌ No JSON found in response")
                structured_content = {"vocabulary": [], "kanji": [], "grammar": [], "phrases": [], "idioms": []}

        # Process all structured content types
        vocab_new = vocab_dup = kanji_new = kanji_dup = grammar_new = grammar_dup = phrase_new = phrase_dup = idiom_new = idiom_dup = 0

        # Process each category (simplified for batch processing)
        for item in structured_content.get("vocabulary", []):
            word = item.get("word", "") if isinstance(item, dict) else ""
            if word and word.strip():
                if db.add_card(word):
                    vocab_new += 1
                else:
                    vocab_dup += 1

        for item in structured_content.get("kanji", []):
            character = item.get("character", "") if isinstance(item, dict) else ""
            if character and character.strip():
                if db.add_kanji(
                    character=character,
                    onyomi=item.get("onyomi", ""),
                    kunyomi=item.get("kunyomi", ""),
                    meanings=item.get("meanings", "")
                ):
                    kanji_new += 1
                else:
                    kanji_dup += 1

        for item in structured_content.get("grammar", []):
            point = item.get("point", "") if isinstance(item, dict) else ""
            if point and point.strip():
                if db.add_grammar(
                    point=point,
                    explanation=item.get("explanation", ""),
                    example=item.get("example", "")
                ):
                    grammar_new += 1
                else:
                    grammar_dup += 1

        for item in structured_content.get("phrases", []):
            phrase = item.get("phrase", "") if isinstance(item, dict) else ""
            if phrase and phrase.strip():
                if db.add_phrase(
                    phrase=phrase,
                    meaning=item.get("meaning", "")
                ):
                    phrase_new += 1
                else:
                    phrase_dup += 1

        for item in structured_content.get("idioms", []):
            idiom = item.get("idiom", "") if isinstance(item, dict) else ""
            if idiom and idiom.strip():
                if db.add_idiom(
                    idiom=idiom,
                    meaning=item.get("meaning", ""),
                    example=item.get("example", "")
                ):
                    idiom_new += 1
                else:
                    idiom_dup += 1

        return vocab_new, kanji_new, grammar_new, phrase_new, idiom_new, vocab_dup, kanji_dup, grammar_dup, phrase_dup, idiom_dup

    def process_discord_log(self) -> None:
        """Process Discord chat logs to extract Japanese content using AI."""
        filepath = self.get_user_input("Enter Discord log file path:")
        if not filepath:
            return

        # Check if file exists
        if not os.path.exists(filepath):
            self.show_message(f"❌ File not found: {filepath}", 4)
            return

        self._draw_base_screen()
        self.stdscr.addstr(4, 2, "🤖 Processing Discord chat log with AI...")
        self.stdscr.refresh()

        # Read file content
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                chat_content = f.read()

            # Use AI to extract Japanese content
            from llm_learn_japanese import structured

            if DEBUG:
                print(f"🤖 Processing Discord chat with AI:")
                print(f"   Chat content length: {len(chat_content)} characters")
                print(f"   Content preview: {chat_content[:200]}...")

            response = self.ai_model.prompt(
                chat_content,
                system=structured.DISCORD_CHAT_PROMPT
            )

            # Parse the JSON response
            response_text = response.text()
            if DEBUG:
                print(f"📤 Discord AI response length: {len(response_text)} characters")
                print(f"📤 Discord AI response preview: {response_text[:300]}...")

            try:
                parsed_content = json.loads(response_text)
                if DEBUG:
                    print(f"✅ Discord JSON parsing successful")
                    print(f"📊 Parsed structure keys: {list(parsed_content.keys()) if isinstance(parsed_content, dict) else 'Not a dict'}")
            except json.JSONDecodeError as e:
                if DEBUG:
                    print(f"❌ Discord JSON parsing failed: {e}")
                parsed_content = {"vocabulary": [], "kanji": [], "grammar": [], "phrases": [], "idioms": []}

            # Add extracted content to database
            vocab_new = kanji_new = grammar_new = phrase_new = idiom_new = 0

            for item in parsed_content.get("vocabulary", []):
                if db.add_card(item.get("word", "")):
                    vocab_new += 1

            for item in parsed_content.get("kanji", []):
                if db.add_kanji(
                    item.get("character", ""),
                    item.get("onyomi", ""),
                    item.get("kunyomi", ""),
                    item.get("meanings", "")
                ):
                    kanji_new += 1

            # Display results
            self._draw_base_screen()
            self.stdscr.addstr(4, 2, "✅ Discord log processed successfully!")
            self.stdscr.addstr(6, 2, f"Added: {vocab_new} vocabulary, {kanji_new} kanji")

        except Exception as e:
            self.stdscr.addstr(4, 2, f"❌ Processing failed: {str(e)}")

        self.stdscr.addstr(10, 2, "Press any key to continue...")
        self.stdscr.refresh()
        self.stdscr.getch()

    def handle_add_content_menu(self) -> None:
        """Handle the add content submenu."""
        options = [
            "Add Vocabulary",
            "Add Kanji",
            "Add Grammar Point",
            "Add Phrase",
            "Add Idiom",
            "Process Image File",
            "Process Multiple Images",
            "Process Discord Log",
            "Back to Main Menu"
        ]

        while True:
            self._draw_base_screen()
            self.draw_menu("Add Content", options, self.menu_index)
            self.stdscr.refresh()

            key = self.stdscr.getch()

            if key == curses.KEY_UP and self.menu_index > 0:
                self.menu_index -= 1
            elif key == curses.KEY_DOWN and self.menu_index < len(options) - 1:
                self.menu_index += 1
            elif key == ord('\n') or key == ord('\r'):
                # Clear any existing message before performing action
                self.message = ""
                if self.menu_index == 0:
                    self.show_add_vocabulary()
                elif self.menu_index == 1:
                    self.show_add_kanji()
                elif self.menu_index == 2:
                    self.show_add_grammar()
                elif self.menu_index == 3:
                    phrase = self.get_user_input("Enter phrase:")
                    if phrase:
                        meaning = self.get_user_input("Enter meaning (optional):") or ""
                        is_new = db.add_phrase(phrase, meaning)
                        if is_new:
                            self.show_message(f"✅ Added phrase: {phrase}", 3)
                        else:
                            self.show_message(f"⚠️ Phrase already exists: {phrase}", 5)
                elif self.menu_index == 4:
                    self.show_add_idiom()
                elif self.menu_index == 5:
                    self.process_image_file()
                elif self.menu_index == 6:
                    self.process_multiple_images()
                elif self.menu_index == 7:
                    self.process_discord_log()
                elif self.menu_index == 8:
                    self.menu_index = 0
                    return
            elif key == ord('q') or key == ord('Q'):
                return
            elif key == ord('b') or key == ord('B'):
                self.menu_index = 0
                return

    def initialize_database(self) -> None:
        """Initialize the database."""
        try:
            if db.is_db_initialized():
                self.show_message("⚠️ Database is already initialized!", 5)
                return

            db.init_db()
            self.show_message("✅ Database initialized successfully!", 3)
        except Exception as e:
            self.show_message(f"❌ Database initialization failed: {str(e)}", 4)

    def set_username(self) -> None:
        """Set the username for progress tracking."""
        username = self.get_user_input(f"Enter username (current: {self.username}):")
        if username:
            self.username = username
            self.show_message(f"✅ Username set to: {self.username}", 3)

    def save_message(self) -> None:
        """Save a message to the database."""
        message = self.get_user_input("Enter message to save:", multiline=True)
        if message:
            try:
                db.save_message(message)
                self.show_message(f"✅ Message saved successfully", 3)
            except Exception as e:
                self.show_message(f"❌ Failed to save message: {str(e)}", 4)

    def manual_card_review(self) -> None:
        """Manually review a card aspect by ID and quality."""
        try:
            aspect_id_str = self.get_user_input("Enter aspect ID:")
            if not aspect_id_str:
                return

            aspect_id = int(aspect_id_str)

            quality_str = self.get_user_input("Enter quality (0-5):\n0=Forgot, 1=Very Poor, 2=Poor, 3=Acceptable, 4=Good, 5=Perfect")
            if not quality_str:
                return

            quality = int(quality_str)

            if quality < 0 or quality > 5:
                self.show_message("❌ Quality must be between 0-5", 4)
                return

            db.review_card(aspect_id, quality)

            if quality >= 3:
                self.show_message(f"✅ Good! Aspect {aspect_id} scheduled for later review.", 3)
            else:
                self.show_message(f"✅ That's okay! Aspect {aspect_id} will be reviewed again soon.", 5)

        except ValueError:
            self.show_message("❌ Please enter valid numbers for aspect ID and quality", 4)
        except Exception as e:
            self.show_message(f"❌ Review failed: {str(e)}", 4)

    def show_user_progress(self) -> None:
        """Show progress for a specific user."""
        username = self.get_user_input("Enter username to view progress:")
        if not username:
            return

        self._draw_base_screen()

        # Get progress for specified user
        progress = db.get_progress(username)

        y = 4
        self.stdscr.attron(curses.color_pair(7) | curses.A_BOLD)
        self.stdscr.addstr(y, 2, f"📊 Progress Statistics for {username}")
        self.stdscr.attroff(curses.color_pair(7) | curses.A_BOLD)

        y += 2
        if progress:
            accuracy = (progress.correct_answers / progress.total_reviews * 100) if progress.total_reviews > 0 else 0

            self.stdscr.addstr(y, 4, f"Total Reviews: {progress.total_reviews}")
            y += 1
            self.stdscr.addstr(y, 4, f"Correct Answers: {progress.correct_answers}")
            y += 1

            # Color code accuracy
            if accuracy >= 80:
                color = 3  # Green
            elif accuracy >= 60:
                color = 5  # Yellow
            else:
                color = 4  # Red

            self.stdscr.attron(curses.color_pair(color))
            self.stdscr.addstr(y, 4, f"Accuracy: {accuracy:.1f}%")
            self.stdscr.attroff(curses.color_pair(color))
            y += 1

            self.stdscr.addstr(y, 4, f"Last Updated: {progress.last_updated.strftime('%Y-%m-%d %H:%M')}")
        else:
            self.stdscr.addstr(y, 4, f"No progress data found for user '{username}'")

        y += 2
        self.stdscr.attron(curses.color_pair(6))
        self.stdscr.addstr(y, 2, "Press any key to continue...")
        self.stdscr.attroff(curses.color_pair(6))
        self.stdscr.refresh()
        self.stdscr.getch()

    def show_ai_status(self) -> None:
        """Show AI configuration status."""
        self._draw_base_screen()

        y = 4
        self.stdscr.attron(curses.color_pair(7) | curses.A_BOLD)
        self.stdscr.addstr(y, 2, "🤖 AI Configuration")
        self.stdscr.attroff(curses.color_pair(7) | curses.A_BOLD)

        y += 2
        self.stdscr.attron(curses.color_pair(3))
        self.stdscr.addstr(y, 4, "✅ OpenAI API Key: Configured")
        self.stdscr.attroff(curses.color_pair(3))
        y += 1
        self.stdscr.addstr(y, 4, f"Model: GPT-5")
        y += 1
        self.stdscr.addstr(y, 4, "AI Features: Active")

        y += 2
        self.stdscr.addstr(y, 2, "AI is powering:")
        y += 1
        self.stdscr.addstr(y, 4, "• Dynamic exercise generation")
        y += 1
        self.stdscr.addstr(y, 4, "• Semantic answer evaluation")
        y += 1
        self.stdscr.addstr(y, 4, "• Intelligent feedback")

        y += 2
        self.stdscr.addstr(y, 2, "Press any key to continue...")
        self.stdscr.refresh()
        self.stdscr.getch()

    def run(self) -> None:
        """Main application loop."""
        # Check and initialize database only if needed
        try:
            if not db.is_db_initialized():
                db.init_db()
                self.show_message("✅ Database initialized on startup", 3)
        except Exception as e:
            self.show_message(f"❌ Database startup check failed: {str(e)}", 4)

        main_options = [
            "📚 Start Study Session",
            "➕ Add Content",
            "📊 View Progress",
            "👥 View User Progress",
            "✍️ Save Message",
            "📝 Manual Card Review",
            "� Set Username",
            "🤖 AI Status",
            "🗄️ Initialize Database",
            "🚪 Exit"
        ]

        while True:
            self._draw_base_screen()
            self.draw_menu("Main Menu", main_options, self.menu_index)
            self.stdscr.refresh()

            key = self.stdscr.getch()

            if key == curses.KEY_UP and self.menu_index > 0:
                self.menu_index -= 1
            elif key == curses.KEY_DOWN and self.menu_index < len(main_options) - 1:
                self.menu_index += 1
            elif key == ord('\n') or key == ord('\r'):
                # Clear any existing message before performing action
                self.message = ""
                if self.menu_index == 0:  # Study Session
                    self.show_study_session()
                elif self.menu_index == 1:  # Add Content
                    old_index = self.menu_index
                    self.menu_index = 0
                    self.handle_add_content_menu()
                    self.menu_index = old_index
                elif self.menu_index == 2:  # View Progress (current user)
                    self.show_progress_stats()
                elif self.menu_index == 3:  # View User Progress (any user)
                    self.show_user_progress()
                elif self.menu_index == 4:  # Save Message
                    self.save_message()
                elif self.menu_index == 5:  # Manual Card Review
                    self.manual_card_review()
                elif self.menu_index == 6:  # Set Username
                    self.set_username()
                elif self.menu_index == 7:  # AI Status
                    self.show_ai_status()
                elif self.menu_index == 8:  # Initialize Database
                    self.initialize_database()
                elif self.menu_index == 9:  # Exit
                    break
            elif key == ord('q') or key == ord('Q'):
                break


def main(stdscr: Any) -> None:
    """Main entry point for the curses application."""
    try:
        app = JapaneseApp(stdscr)
        app.run()
    except Exception as e:
        # Show error in a safe way
        stdscr.clear()
        stdscr.addstr(0, 0, f"Error: {str(e)}")
        stdscr.addstr(2, 0, "Traceback:")
        error_lines = traceback.format_exc().split('\n')
        for i, line in enumerate(error_lines[:10]):  # Show first 10 lines
            stdscr.addstr(3 + i, 0, line[:80])  # Truncate long lines
        stdscr.addstr(15, 0, "Press any key to exit...")
        stdscr.refresh()
        stdscr.getch()


if __name__ == "__main__":
    try:
        curses.wrapper(main)
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)