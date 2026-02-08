"""add repetitions column to card_aspects

Revision ID: a1b2c3d4e5f6
Revises: 2d179958d92a
Create Date: 2026-02-08 01:23:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, Sequence[str], None] = '2d179958d92a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add repetitions column to card_aspects table (SM-2 repetition counter)."""
    op.add_column('card_aspects', sa.Column('repetitions', sa.Integer(), nullable=False, server_default='0'))


def downgrade() -> None:
    """Remove repetitions column from card_aspects table."""
    op.drop_column('card_aspects', 'repetitions')
