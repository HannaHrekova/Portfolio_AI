import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from tools import alfred, system_message_content

def chat_interface(message: str, history: list) -> str:
    
    langchain_messages = [SystemMessage(content=system_message_content)]
    
    for msg in history:
        if isinstance(msg, dict):
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "user" and content:
                langchain_messages.append(HumanMessage(content=content))
            elif role == "assistant" and content:
                langchain_messages.append(AIMessage(content=content))
        elif isinstance(msg, (list, tuple)) and len(msg) == 2:
            human_msg, ai_msg = msg
            if human_msg:
                langchain_messages.append(HumanMessage(content=human_msg))
            if ai_msg:
                langchain_messages.append(AIMessage(content=ai_msg))
    
    langchain_messages.append(HumanMessage(content=message))
    
    print(f"DEBUG: Messages sent to Alfred: {langchain_messages}")
    
    result = alfred.invoke({"messages": langchain_messages})
    final_ai_message = result["messages"][-1]
    
    if isinstance(final_ai_message, AIMessage):
        print(f"DEBUG: Final AI Message Content: '{final_ai_message.content}'")
        return final_ai_message.content
    elif hasattr(final_ai_message, "tool_calls") and final_ai_message.tool_calls:
        print(f"DEBUG: Agent ended with ToolCalls: {final_ai_message.tool_calls}")
        return "Alfred seems to have completed the tool challenge, but has not formulated a final answer. Try to clarify the question."
    else:
        print(f"DEBUG: Agent ended with unexpected message type: {type(final_ai_message)}. Content: {getattr(final_ai_message, 'content', '')}")
        return "Alfred completed the work, but did not return the expected response."

iface = gr.ChatInterface(
    fn=chat_interface, 
    title="🎩 Alfred - Your Gala Assistant"
)

if __name__ == "__main__":
    iface.launch()
