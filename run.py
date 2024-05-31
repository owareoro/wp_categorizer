import mysql.connector
from time import time
from train import (
    classify_post,
    config,
    load_model,
    parallel_map,
    CONFIG,
    create_defaultdict,
    use_buffer,
)


def load_categories():
    for id, name, category_type in fetch_categories():
        display_name = "*" + name if category_type == "category" else name
        categories[display_name] = id
        categories[id] = display_name
    return categories


@use_buffer(".categories.temp")
def fetch_categories():
    import mysql.connector

    db = None
    try:
        db = mysql.connector.connect(
            host=config.get("Settings", "Host"),
            port=config.get("Settings", "Port"),
            user=config.get("Settings", "User"),
            passwd=config.get("Settings", "Password"),
            database=config.get("Settings", "Database"),
            connect_timeout=60000,
        )
        cursor = db.cursor()
        cursor.execute(
            """
            SELECT wp_term_taxonomy.term_taxonomy_id, name, taxonomy
            FROM wp_terms
            JOIN wp_term_taxonomy ON wp_term_taxonomy.term_id = wp_terms.term_id 
            WHERE taxonomy = "category" OR taxonomy = "post_tag"
        """
        )
        yield from cursor.fetchall()
    except mysql.connector.Error as err:
        print(f"Error: {err}")
    finally:
        if db and db.is_connected():
            cursor.close()
            db.close()


@use_buffer(".buffer.temp")
def fetch_posts():
    db = None
    try:
        db = mysql.connector.connect(
            host=config.get("Settings", "Host"),
            port=config.get("Settings", "Port"),
            user=config.get("Settings", "User"),
            passwd=config.get("Settings", "Password"),
            database=config.get("Settings", "Database"),
            connect_timeout=60000,
        )
        db._execute_query("SET net_read_timeout=31536000;")
        db._execute_query("SET GLOBAL max_allowed_packet=1073741824;")
        cursor = db.cursor()
        cursor.execute(
            """
            SELECT ID, post_content
            FROM wp_posts
            LEFT JOIN wp_term_relationships ON wp_posts.ID = wp_term_relationships.object_id 
            WHERE post_type = 'post' AND term_taxonomy_id is NULL
            ORDER BY wp_posts.ID;
        """
        )
        while True:
            print("Reading 1000 rows...")
            chunk = cursor.fetchmany(size=1000)
            if not chunk:
                break
            print("Done")
            for row in chunk:
                yield row

    except mysql.connector.Error as err:
        print(f"Error: {err}")
    finally:
        if db and db.is_connected():
            cursor.close()
            db.close()


def run_classify(id, content, model):
    if len(content) < 5:
        return id, []
    results = classify_post(content, model, **CONFIG)
    return id, results


categories = {}
if __name__ == "__main__":
    model = load_model()
    categories = load_categories()
    i = 0
    start = time()

    with open("result.sql", "w") as f:
        f.write(
            "INSERT INTO wp_term_relationships (object_id, term_taxonomy_id) VALUES\n"
        )
        for id, results in parallel_map(run_classify, fetch_posts(), model):
            i = i + 1
            if i % 20 == 0:
                print(f"Wrote {i} {time() - start} seconds")
            for category in results:
                f.write(f"({id}, {categories[category]})\n")
        f.write("ON DUPLICATE KEY IGNORE;")
