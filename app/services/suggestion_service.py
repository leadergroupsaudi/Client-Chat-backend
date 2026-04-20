from app.llm_providers.groq_provider import generate_response
from sqlalchemy.orm import Session

async def get_suggested_replies(db: Session, company_id: int, conversation_history: list[str]) -> list[str]:
    """
    Generates a list of suggested replies based on the conversation history.
    """

    prompt = f"Based on the following conversation, please provide a list of 3 suggested replies for the agent:\n\n{''}".join(conversation_history)
    
    response = await generate_response(db=db, company_id=company_id, model_name="mixtral-8x7b-32768", system_prompt="You are a helpful assistant.", chat_history=[{"role": "user", "content": prompt}])
    return response["content"].split('\n')
