"""add top_kanji and top_words tables

Revision ID: c2d3e4f5a6b7
Revises: b1c2d3e4f5a6
Create Date: 2026-02-08 02:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c2d3e4f5a6b7'
down_revision: Union[str, Sequence[str], None] = 'b1c2d3e4f5a6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create top_kanji and top_words tables."""
    op.create_table('top_kanji',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('rank', sa.Integer(), nullable=False),
        sa.Column('kanji', sa.String(), nullable=False),
        sa.Column('count', sa.Integer(), nullable=True),
        sa.Column('on_readings', sa.Text(), nullable=True),
        sa.Column('kun_readings', sa.Text(), nullable=True),
        sa.Column('meanings_en', sa.Text(), nullable=True),
        sa.Column('jlpt_level', sa.String(), nullable=True),
        sa.Column('annotated', sa.Boolean(), nullable=True, server_default='0'),
        sa.Column('next_review', sa.DateTime(), nullable=True),
        sa.Column('interval', sa.Integer(), nullable=True, server_default='1'),
        sa.Column('ease_factor', sa.Float(), nullable=True, server_default='2.5'),
        sa.Column('repetitions', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('success_count', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('lesson_viewed', sa.Boolean(), nullable=True, server_default='0'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('rank')
    )

    op.create_table('top_words',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('rank', sa.Integer(), nullable=False),
        sa.Column('lemma', sa.String(), nullable=False),
        sa.Column('reading_csv', sa.String(), nullable=True),
        sa.Column('count', sa.Integer(), nullable=True),
        sa.Column('reading', sa.Text(), nullable=True),
        sa.Column('meanings_en', sa.Text(), nullable=True),
        sa.Column('pos_tags', sa.Text(), nullable=True),
        sa.Column('jlpt_level', sa.String(), nullable=True),
        sa.Column('annotated', sa.Boolean(), nullable=True, server_default='0'),
        sa.Column('next_review', sa.DateTime(), nullable=True),
        sa.Column('interval', sa.Integer(), nullable=True, server_default='1'),
        sa.Column('ease_factor', sa.Float(), nullable=True, server_default='2.5'),
        sa.Column('repetitions', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('success_count', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('lesson_viewed', sa.Boolean(), nullable=True, server_default='0'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('rank')
    )


def downgrade() -> None:
    """Drop top_kanji and top_words tables."""
    op.drop_table('top_words')
    op.drop_table('top_kanji')
