"""add bunpro_grammar table

Revision ID: b1c2d3e4f5a6
Revises: a1b2c3d4e5f6
Create Date: 2026-02-08 01:40:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b1c2d3e4f5a6'
down_revision: Union[str, Sequence[str], None] = 'a1b2c3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create bunpro_grammar table."""
    op.create_table('bunpro_grammar',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('rank', sa.Integer(), nullable=False),
        sa.Column('usage_tier', sa.String(), nullable=False),
        sa.Column('score', sa.Float(), nullable=True),
        sa.Column('register', sa.String(), nullable=True),
        sa.Column('jlpt', sa.String(), nullable=True),
        sa.Column('bunpro_id', sa.Float(), nullable=True),
        sa.Column('grammar', sa.String(), nullable=False),
        sa.Column('meaning', sa.Text(), nullable=True),
        sa.Column('url', sa.String(), nullable=True),
        sa.Column('norm', sa.String(), nullable=True),
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
    """Drop bunpro_grammar table."""
    op.drop_table('bunpro_grammar')
