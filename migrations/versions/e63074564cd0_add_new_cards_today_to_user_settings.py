"""add new_cards_today to user_settings

Revision ID: e63074564cd0
Revises: 275dc3c343cd
Create Date: 2025-09-27 15:07:53.125663

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e63074564cd0'
down_revision: Union[str, Sequence[str], None] = '275dc3c343cd'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("user_settings", sa.Column("new_cards_today", sa.Integer(), nullable=False, server_default="0"))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("user_settings", "new_cards_today")
