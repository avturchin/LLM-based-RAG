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

class GeminiFileProcessor:
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash-exp"):
        self.api_key = api_key
        self.model = model
        # Используем tiktoken для подсчета токенов (примерная оценка для Gemini)
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.max_tokens_per_chunk = 200_000
        
    def count_tokens(self, text: str) -> int:
        """Подсчет токенов в тексте"""
        return len(self.encoding.encode(text))
    
    def split_into_chunks(self, text: str, prompt: str) -> List[str]:
        """Разбивка текста на чанки по 200 тысяч токенов"""
        prompt_tokens = self.count_tokens(prompt)
        available_tokens = self.max_tokens_per_chunk - prompt_tokens - 2000  # резерв для ответа
        
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
        """Обработка одного чанка через Gemini API"""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        
        headers = {
            "Content-Type": "application/json"
        }
        
        # Формируем запрос для Gemini API
        data = {
            "contents": [
                {
                    "parts": [
                        {"text": f"{prompt}\n\nТекст для обработки:\n{chunk}"}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.7,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 8192,
                "candidateCount": 1
            },
            "safetySettings": [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                }
            ]
        }
        
        try:
            async with session.post(url, headers=headers, json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    
                    # Извлекаем содержимое ответа
                    if "candidates" in result and len(result["candidates"]) > 0:
                        candidate = result["candidates"][0]
                        if "content" in candidate and "parts" in candidate["content"]:
                            content = candidate["content"]["parts"][0]["text"]
                            
                            # Подсчитываем токены (примерно)
                            input_tokens = self.count_tokens(chunk + prompt)
                            output_tokens = self.count_tokens(content)
                            total_tokens = input_tokens + output_tokens
                            
                            print(f"✓ Обработан чанк {chunk_index + 1}")
                            return {
                                "chunk_index": chunk_index,
                                "success": True,
                                "content": content,
                                "tokens_used": total_tokens,
                                "input_tokens": input_tokens,
                                "output_tokens": output_tokens
                            }
                        else:
                            print(f"✗ Пустой ответ для чанка {chunk_index + 1}")
                            return {
                                "chunk_index": chunk_index,
                                "success": False,
                                "error": "Пустой ответ от Gemini"
                            }
                    else:
                        print(f"✗ Нет кандидатов в ответе для чанка {chunk_index + 1}")
                        return {
                            "chunk_index": chunk_index,
                            "success": False,
                            "error": "Нет кандидатов в ответе"
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
    
    async def process_file(self, file_path: str, prompt: str, output_path: str):
        """Основная функция обработки файла"""
        print(f"Загрузка файла: {file_path}")
        
        # Чтение файла
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"Размер файла: {len(content)} символов")
        print(f"Примерное количество токенов в файле: {self.count_tokens(content)}")
        
        # Разбивка на чанки
        chunks = self.split_into_chunks(content, prompt)
        print(f"Файл разбит на {len(chunks)} чанков по ~{self.max_tokens_per_chunk:,} токенов")
        
        # Обработка чанков с задержкой для соблюдения rate limits
        results = []
        connector = aiohttp.TCPConnector(limit=10)  # Ограничиваем количество соединений
        timeout = aiohttp.ClientTimeout(total=300)  # Увеличиваем таймаут до 5 минут
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            # Обрабатываем чанки с небольшой задержкой
            for i, chunk in enumerate(chunks):
                print(f"Обработка чанка {i + 1}/{len(chunks)}...")
                result = await self.process_chunk(session, chunk, prompt, i)
                results.append(result)
                
                # Задержка между запросами для соблюдения rate limits Gemini
                if i < len(chunks) - 1:  # Не ждем после последнего чанка
                    await asyncio.sleep(1)  # 1 секунда между запросами
        
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
        
        # Статистика
        total_tokens = sum(r.get("tokens_used", 0) for r in successful_results)
        total_input_tokens = sum(r.get("input_tokens", 0) for r in successful_results)
        total_output_tokens = sum(r.get("output_tokens", 0) for r in successful_results)
        
        print(f"\n✓ Обработка завершена!")
        print(f"  Успешно обработано: {len(successful_results)}/{len(chunks)} чанков")
        print(f"  Общее количество токенов: {total_tokens:,}")
        print(f"  Входящие токены: {total_input_tokens:,}")
        print(f"  Исходящие токены: {total_output_tokens:,}")
        print(f"  Результат сохранен в: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Обработка больших файлов через Gemini API")
    parser.add_argument("--file", required=True, help="Путь к входному файлу")
    parser.add_argument("--prompt", required=True, help="Промпт для обработки")
    parser.add_argument("--output", required=True, help="Путь к выходному файлу")
    parser.add_argument("--model", default="gemini-2.0-flash-exp", 
                       help="Модель Gemini (по умолчанию: gemini-2.0-flash-exp)")
    parser.add_argument("--api-key", help="API ключ Gemini (или используйте переменную GEMINI_API_KEY)")
    
    args = parser.parse_args()
    
    # Получение API ключа
    api_key = args.api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ Не указан API ключ Gemini. Используйте --api-key или установите переменную GEMINI_API_KEY")
        print("Получить API ключ можно на: https://aistudio.google.com/app/apikey")
        sys.exit(1)
    
    # Проверка существования входного файла
    if not os.path.exists(args.file):
        print(f"❌ Файл не найден: {args.file}")
        sys.exit(1)
    
    # Создание процессора и запуск
    processor = GeminiFileProcessor(api_key, args.model)
    
    try:
        asyncio.run(processor.process_file(args.file, args.prompt, args.output))
    except KeyboardInterrupt:
        print("\n⚠️  Обработка прервана пользователем")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
