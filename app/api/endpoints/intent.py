from fastapi import APIRouter, HTTPException, status
from app.schemas.intent import IntentsData, ImportSummary
from app.config.database import create_connection, create_tables
import mysql.connector

router = APIRouter()


@router.post("/import-intents", response_model=ImportSummary)
async def import_intents(data: IntentsData):
    """Import intents data from JSON to database."""
    try:
        conn = create_connection()
        cursor = conn.cursor(dictionary=True)

        # Create tables if they don't exist
        await create_tables(cursor)

        # Initialize counters

        new_intents_count = 0

        new_patterns_count = 0

        new_responses_count = 0

        updated_intents_count = 0

        # Import data

        for intent in data.intents:

            # Check if intent already exists

            cursor.execute("SELECT id FROM intents WHERE tag = %s", (intent.tag,))

            existing_intent = cursor.fetchone()

            if existing_intent:

                # Intent exists, update patterns and responses

                intent_id = existing_intent["id"]

                updated_intents_count += 1

                # Get existing patterns

                cursor.execute(
                    "SELECT pattern FROM patterns WHERE intent_id = %s", (intent_id,)
                )

                existing_patterns = {row["pattern"] for row in cursor.fetchall()}

                # Add new patterns

                for pattern in intent.patterns:

                    if pattern not in existing_patterns:

                        cursor.execute(
                            "INSERT INTO patterns (intent_id, pattern) VALUES (%s, %s)",
                            (intent_id, pattern),
                        )

                        new_patterns_count += 1

                # Get existing responses

                cursor.execute(
                    "SELECT response FROM responses WHERE intent_id = %s", (intent_id,)
                )

                existing_responses = {row["response"] for row in cursor.fetchall()}

                # Add new responses

                for response in intent.responses:

                    if response not in existing_responses:

                        cursor.execute(
                            "INSERT INTO responses (intent_id, response) VALUES (%s, %s)",
                            (intent_id, response),
                        )

                        new_responses_count += 1

            else:

                # Insert new intent

                cursor.execute("INSERT INTO intents (tag) VALUES (%s)", (intent.tag,))

                intent_id = cursor.lastrowid

                new_intents_count += 1

                # Insert all patterns

                for pattern in intent.patterns:

                    cursor.execute(
                        "INSERT INTO patterns (intent_id, pattern) VALUES (%s, %s)",
                        (intent_id, pattern),
                    )

                    new_patterns_count += 1

                # Insert all responses

                for response in intent.responses:

                    cursor.execute(
                        "INSERT INTO responses (intent_id, response) VALUES (%s, %s)",
                        (intent_id, response),
                    )

                    new_responses_count += 1

        # Commit changes

        conn.commit()

        return ImportSummary(
            status="success",
            new_intents_count=new_intents_count,
            new_patterns_count=new_patterns_count,
            new_responses_count=new_responses_count,
            updated_intents_count=updated_intents_count,
            message="Data imported successfully!",
        )
    except mysql.connector.Error as e:

        if "conn" in locals() and conn.is_connected():

            conn.rollback()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}",
        )

    except Exception as e:
        if "conn" in locals() and conn.is_connected():
            conn.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error importing data: {str(e)}",
        )
    finally:
        if "conn" in locals() and conn.is_connected():
            cursor.close()
            conn.close()


@router.get("/get-intents", response_model=IntentsData)
async def get_intents():
    """Retrieve all intents data from database."""
    try:
        conn = create_connection()
        cursor = conn.cursor(dictionary=True)

        intents_list = []

        # Get all intents

        cursor.execute("SELECT * FROM intents")

        intents = cursor.fetchall()

        for intent in intents:

            # Get patterns for this intent

            cursor.execute(
                "SELECT pattern FROM patterns WHERE intent_id = %s", (intent["id"],)
            )

            patterns = [row["pattern"] for row in cursor.fetchall()]

            # Get responses for this intent

            cursor.execute(
                "SELECT response FROM responses WHERE intent_id = %s", (intent["id"],)
            )

            responses = [row["response"] for row in cursor.fetchall()]

            intents_list.append(
                {"tag": intent["tag"], "patterns": patterns, "responses": responses}
            )

        return IntentsData(intents=intents_list)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving intents: {str(e)}",
        )
    finally:
        if "conn" in locals() and conn.is_connected():
            cursor.close()
            conn.close()
