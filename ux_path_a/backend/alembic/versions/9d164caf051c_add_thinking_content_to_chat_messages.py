"""add_thinking_content_to_chat_messages

Revision ID: 9d164caf051c
Revises: 
Create Date: 2026-01-11 18:53:55.216501

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


# revision identifiers, used by Alembic.
revision = '9d164caf051c'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Check if table exists
    conn = op.get_bind()
    inspector = inspect(conn)
    table_exists = 'chat_messages' in inspector.get_table_names()
    
    if not table_exists:
        # Table doesn't exist - create it with all columns including thinking_content
        # First ensure dependent tables exist
        if 'users' not in inspector.get_table_names():
            op.create_table(
                'users',
                sa.Column('id', sa.Integer(), nullable=False),
                sa.Column('email', sa.String(), nullable=False),
                sa.Column('username', sa.String(), nullable=False),
                sa.Column('hashed_password', sa.String(), nullable=False),
                sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
                sa.Column('updated_at', sa.DateTime(timezone=True)),
                sa.Column('is_active', sa.Boolean(), default=True),
                sa.PrimaryKeyConstraint('id'),
                sa.UniqueConstraint('email'),
                sa.UniqueConstraint('username')
            )
        
        if 'chat_sessions' not in inspector.get_table_names():
            op.create_table(
                'chat_sessions',
                sa.Column('id', sa.String(), nullable=False),
                sa.Column('user_id', sa.Integer(), nullable=False),
                sa.Column('title', sa.String(), nullable=True),
                sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
                sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
                sa.Column('total_tokens_used', sa.Integer(), default=0),
                sa.PrimaryKeyConstraint('id'),
                sa.ForeignKeyConstraint(['user_id'], ['users.id'])
            )
        
        # Now create chat_messages table
        op.create_table(
            'chat_messages',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('session_id', sa.String(), nullable=False),
            sa.Column('role', sa.String(), nullable=False),
            sa.Column('content', sa.Text(), nullable=False),
            sa.Column('thinking_content', sa.Text(), nullable=True),
            sa.Column('tool_calls', sa.JSON(), nullable=True),
            sa.Column('tool_results', sa.JSON(), nullable=True),
            sa.Column('token_usage', sa.JSON(), nullable=True),
            sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
            sa.PrimaryKeyConstraint('id'),
            sa.ForeignKeyConstraint(['session_id'], ['chat_sessions.id'])
        )
    else:
        # Table exists - just add the column if it doesn't exist
        columns = [col['name'] for col in inspector.get_columns('chat_messages')]
        if 'thinking_content' not in columns:
            op.add_column('chat_messages', sa.Column('thinking_content', sa.Text(), nullable=True))


def downgrade() -> None:
    # Check if table exists
    conn = op.get_bind()
    inspector = inspect(conn)
    
    if 'chat_messages' in inspector.get_table_names():
        columns = [col['name'] for col in inspector.get_columns('chat_messages')]
        if 'thinking_content' in columns:
            op.drop_column('chat_messages', 'thinking_content')
