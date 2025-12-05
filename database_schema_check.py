"""Utility to check pgvector dimensionality vs config.EMBEDDING_DIM."""
import sys
from database import Database
from config import EMBEDDING_DIM
from sqlalchemy import text


def main():
    db = Database()
    if db.use_stub:
        print("Stub DB: schema check not applicable.")
        return
    engine = db._engine
    assert engine is not None
    with engine.begin() as conn:
        res = conn.execute(text(
            """
            SELECT t.relname AS table_name, a.attname AS column_name, atttypmod
            FROM pg_attribute a
            JOIN pg_class t ON a.attrelid = t.oid
            WHERE t.relname = 'articles' AND a.attname = 'embedding'
            """
        ))
        row = res.fetchone()
        if not row:
            print("embedding column not found; run schema creation.")
            sys.exit(1)
        # pgvector stores typmod = dim + 4
        typmod = row[2]
        dim = typmod - 4 if typmod else None
        if dim != EMBEDDING_DIM:
            print(f"Mismatch: embedding dim in DB={dim}, config={EMBEDDING_DIM}. Drop/alter table to proceed.")
            sys.exit(2)
        print("embedding dimension matches config.")


if __name__ == "__main__":
    main()
