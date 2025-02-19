import os
from ingest import Ingest
from chat import Chat

chat = Chat()
ingest = Ingest()

def main():
    while (True):
        query = input("\n🚀 Câu hỏi:\n")
        
        if query.lower() in ["exit", "quit", "q"]:
            break

        if (query.lower() in ["ingest", "i"]):
            files = os.listdir('./data')
            for f in files:
                filepath = os.path.join('./data', f)
                if os.path.exists(filepath):
                    print(f"🚀 Ingesting {f}...")
                    ingest.ingest_doc(filepath)
                    print(f"🚀 Ingest {f} thành công")
            continue

        result = chat.ask(query)
        print(f"🤖 Trả lời:\n{result}")

if __name__ == "__main__":
    main()
