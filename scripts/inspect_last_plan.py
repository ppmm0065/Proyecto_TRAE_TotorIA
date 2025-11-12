import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'seguimiento.db')

def main():
    db_path = os.path.abspath(DB_PATH)
    print(f"DB path: {db_path}")
    print("DB exists:", os.path.exists(db_path))
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, timestamp, related_entity_type, related_entity_name, follow_up_comment
        FROM follow_ups
        WHERE follow_up_type='intervention_plan'
        ORDER BY id DESC
        LIMIT 1
        """
    )
    row = cur.fetchone()
    if not row:
        print("No intervention plan rows found")
        return
    print("ID:", row["id"], "ts:", row["timestamp"], "entity:", row["related_entity_type"], row["related_entity_name"])
    txt = row["follow_up_comment"] or ""
    print("Length:", len(txt))
    print("--- RAW MARKDOWN START ---")
    print(txt)
    print("--- RAW MARKDOWN END ---")

if __name__ == "__main__":
    main()

