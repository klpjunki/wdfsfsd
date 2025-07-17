from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'ebf08581de14'
down_revision: Union[str, None] = '3ac79871ed47'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade():
    op.add_column('users', sa.Column('daily_streak', sa.Integer(), nullable=False, server_default="0"))
    op.add_column('users', sa.Column('last_daily_login', sa.DateTime(), nullable=True))
    op.add_column('users', sa.Column('role', sa.String(length=20), nullable=False))
    op.add_column('users', sa.Column('tg_subscribed', sa.Boolean(), nullable=False, server_default=sa.text("0")))

def downgrade():
    op.drop_column('users', 'tg_subscribed')
    op.drop_column('users', 'last_daily_login')
    op.drop_column('users', 'daily_streak')
    op.drop_column('users', 'role')
