"""add_communication_provider_to_widget_settings

Adds a communication_provider column to the widget_settings table to allow
each agent widget to specify which voice provider to use:
  - 'livekit'         : existing LiveKit-based voice agents (default)
  - 'openai_realtime' : new OpenAI Realtime API voice (no LiveKit)

Revision ID: h7i8j9k0l1m2
Revises: g6h7i8j9k0l1
Create Date: 2026-04-08

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "h7i8j9k0l1m2"
down_revision: Union[str, Sequence[str], None] = "g6h7i8j9k0l1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add communication_provider column with default 'livekit'."""
    op.add_column(
        "widget_settings",
        sa.Column(
            "communication_provider",
            sa.String(),
            nullable=True,
            server_default="livekit",
        ),
    )


def downgrade() -> None:
    """Remove communication_provider column."""
    op.drop_column("widget_settings", "communication_provider")
