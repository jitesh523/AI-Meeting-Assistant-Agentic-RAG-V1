"""
Add similarity scores to rag_results and enable pg_trgm with index on documents.text

Revision ID: 20251104_add_rag_similarity_pgtrgm
Revises: 
Create Date: 2025-11-04
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '20251104_add_rag_similarity_pgtrgm'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    # Enable pg_trgm extension
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")

    # Add similarity_scores column to rag_results if not exists
    op.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'rag_results' AND column_name = 'similarity_scores'
            ) THEN
                ALTER TABLE rag_results ADD COLUMN similarity_scores JSONB;
            END IF;
        END$$;
        """
    )

    # Add trigram index for lexical search on documents.text
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_documents_text_trgm
        ON documents USING gin (text gin_trgm_ops);
        """
    )


def downgrade():
    # Drop index (keep extension and column to avoid data loss during downgrade)
    op.execute("DROP INDEX IF EXISTS idx_documents_text_trgm;")
    # Optionally remove column
    # op.execute("ALTER TABLE rag_results DROP COLUMN IF EXISTS similarity_scores;")
