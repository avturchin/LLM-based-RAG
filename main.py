# main.py
import os
import sys
import json
import asyncio
import aiohttp
import tiktoken
from typing import List, Optional
import argparse
from pathlib import Path

class LLMFileProcessor:
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.model = model
        self.encoding = tiktoken.encoding_for_model(model)
        self.max_tokens_per_chunk = 1_000_000
        
    def count_tokens(self, text: str) -> int:
        """Подсчет токенов в тексте"""
        return len(self.encoding.encode(text))
    
    def split_into_chunks(self, text: str, prompt: str) -> List[str]:
        """Разбивка текста на чанки по 1 млн токенов"""
        prompt_tokens = self.count_tokens(prompt)
        available_tokens = self.max_tokens_per_chunk - prompt_tokens - 500  # резерв для ответа
        
        chunks = []
        current_chunk = ""
        
        # Разбиваем по абзацам для лучшего контекста
        paragraphs = text.split('\n\n')
        
        for paragraph in paragraphs:
            paragraph_tokens = self.count_tokens(paragraph)
            current_chunk_tokens = self.count_tokens(current_chunk)
            
            if current_chunk_tokens + paragraph_tokens > available_tokens:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = paragraph
                else:
                    # Если один абзац больше лимита, разбиваем по предложениям
                    sentences = paragraph.split('. ')
                    for sentence in sentences:
                        if self.count_tokens(current_chunk + sentence) > available_tokens:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                                current_chunk = sentence
                            else:
                                chunks.append(sentence.strip())
                        else:
                            current_chunk += sentence + ". "
            else:
                current_chunk += paragraph + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks
    
    async def process_chunk(self, session: aiohttp.ClientSession, chunk: str, prompt: str, chunk_index: int) -> dict:
        """Обработка одного чанка через LLM API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": chunk}
            ],
            "temperature": 0.7
        }
        
        try:
            async with session.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    content = result["choices"][0]["message"]["content"]
                    print(f"✓ Обработан чанк {chunk_index + 1}")
                    return {
                        "chunk_index": chunk_index,
                        "success": True,
                        "content": content,
                        "tokens_used": result.get("usage", {}).get("total_tokens", 0)
                    }
                else:
                    error_text = await response.text()
                    print(f"✗ Ошибка в чанке {chunk_index + 1}: {response.status} - {error_text}")
                    return {
                        "chunk_index": chunk_index,
                        "success": False,
                        "error": f"HTTP {response.status}: {error_text}"
                    }
        except Exception as e:
            print(f"✗ Исключение в чанке {chunk_index + 1}: {str(e)}")
            return {
                "chunk_index": chunk_index,
                "success": False,
                "error": str(e)
            }
    
    async def process_file(self, file_path: str, prompt: str, output_path: str):
        """Основная функция обработки файла"""
        print(f"Загрузка файла: {file_path}")
        
        # Чтение файла
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"Размер файла: {len(content)} символов")
        print(f"Токенов в файле: {self.count_tokens(content)}")
        
        # Разбивка на чанки
        chunks = self.split_into_chunks(content, prompt)
        print(f"Файл разбит на {len(chunks)} чанков")
        
        # Обработка чанков
        results = []
        async with aiohttp.ClientSession() as session:
            tasks = [
                self.process_chunk(session, chunk, prompt, i)
                for i, chunk in enumerate(chunks)
            ]
            results = await asyncio.gather(*tasks)
        
        # Сбор результатов
        successful_results = [r for r in results if r["success"]]
        failed_results = [r for r in results if not r["success"]]
        
        if failed_results:
            print(f"⚠️  Не удалось обработать {len(failed_results)} чанков")
            for failed in failed_results:
                print(f"   Чанк {failed['chunk_index'] + 1}: {failed['error']}")
        
        # Сохранение результатов
        final_content = "\n\n".join([
            f"=== РЕЗУЛЬТАТ ЧАНКА {r['chunk_index'] + 1} ===\n{r['content']}"
            for r in sorted(successful_results, key=lambda x: x['chunk_index'])
        ])
        
        # Создание директории если не существует
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_content)
        
        total_tokens = sum(r.get("tokens_used", 0) for r in successful_results)
        print(f"\n✓ Обработка завершена!")
        print(f"  Успешно обработано: {len(successful_results)}/{len(chunks)} чанков")
        print(f"  Общее количество токенов: {total_tokens}")
        print(f"  Результат сохранен в: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Обработка больших файлов через LLM")
    parser.add_argument("--file", required=True, help="Путь к входному файлу")
    parser.add_argument("--prompt", required=True, help="Промпт для обработки")
    parser.add_argument("--output", required=True, help="Путь к выходному файлу")
    parser.add_argument("--model", default="gpt-3.5-turbo", help="Модель LLM")
    parser.add_argument("--api-key", help="API ключ (или используйте переменную OPENAI_API_KEY)")
    
    args = parser.parse_args()
    
    # Получение API ключа
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Не указан API ключ. Используйте --api-key или установите переменную OPENAI_API_KEY")
        sys.exit(1)
    
    # Проверка существования входного файла
    if not os.path.exists(args.file):
        print(f"❌ Файл не найден: {args.file}")
        sys.exit(1)
    
    # Создание процессора и запуск
    processor = LLMFileProcessor(api_key, args.model)
    
    try:
        asyncio.run(processor.process_file(args.file, args.prompt, args.output))
    except KeyboardInterrupt:
        print("\n⚠️  Обработка прервана пользователем")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
