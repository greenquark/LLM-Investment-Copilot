"""add_thinking_content_to_chat_messages

Revision ID: 9d164caf051c
Revises: 
Create Date: 2026-01-11 18:53:55.216501

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '9d164caf051c'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('chat_messages', sa.Column('thinking_content', sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column('chat_messages', 'thinking_content')
