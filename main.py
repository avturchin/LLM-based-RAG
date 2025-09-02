import os
import sys
import json
import asyncio
import aiohttp
import tiktoken
from typing import List, Optional
import argparse
from pathlib import Path
import time
from datetime import datetime

class GeminiFileProcessor:
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash", paid_tier: bool = False):
        self.api_key = api_key
        self.model = model
        self.paid_tier = paid_tier
        self.encoding = tiktoken.get_encoding("cl100k_base")
        
        # Лимиты для разных уровней и моделей
        if paid_tier:
            self.rate_limits = {
                "gemini-2.0-flash": {"rpm": 2000, "tpm": 4000000, "rpd": 10000000},
                "gemini-2.0-flash-lite": {"rpm": 4000, "tpm": 4000000, "rpd": 10000000},
                "gemini-2.5-flash": {"rpm": 1000, "tpm": 1000000, "rpd": 10000},
                "gemini-2.5-flash-lite": {"rpm": 4000, "tpm": 4000000, "rpd": 10000000},
                "gemini-2.5-pro": {"rpm": 150, "tpm": 2000000, "rpd": 10000},
                "gemini-1.5-flash": {"rpm": 2000, "tpm": 4000000, "rpd": None},
                "gemini-1.5-pro": {"rpm": 1000, "tpm": 4000000, "rpd": None}
            }
        else:
            # Free Tier лимиты
            self.rate_limits = {
                "gemini-2.0-flash": {"rpm": 15, "tpm": 1000000, "rpd": 200},
                "gemini-2.0-flash-lite": {"rpm": 30, "tpm": 1000000, "rpd": 200},
                "gemini-2.5-flash": {"rpm": 10, "tpm": 250000, "rpd": 250},
                "gemini-2.5-flash-lite": {"rpm": 15, "tpm": 250000, "rpd": 1000},
                "gemini-2.5-pro": {"rpm": 5, "tpm": 250000, "rpd": 100},
                "gemini-1.5-flash": {"rpm": 15, "tpm": 250000, "rpd": 50},
                "gemini-1.5-pro": {"rpm": 2, "tpm": 32000, "rpd": 50}
            }
        
        self.current_limits = self.rate_limits.get(model, {"rpm": 15, "tpm": 250000, "rpd": 200})
        
        tier_name = "Платный" if paid_tier else "Free"
        print(f"🤖 Модель: {model}")
        print(f"💳 Уровень: {tier_name}")
        print(f"📊 Лимиты: {self.current_limits['rpm']} RPM, {self.current_limits['tpm']:,} TPM")
        
    def count_tokens(self, text: str) -> int:
        """Подсчет токенов в тексте"""
        return len(self.encoding.encode(text))
    
    def split_into_chunks(self, text: str, prompt: str, max_chunk_tokens: int) -> List[str]:
        """Разбивка текста на чанки с учетом лимитов и минимального размера 10,000 токенов"""
        prompt_tokens = self.count_tokens(prompt)
        available_tokens = max_chunk_tokens - prompt_tokens - 2000  # резерв для ответа
        
        # Устанавливаем минимальный размер чанка в 10,000 токенов
        min_chunk_tokens = 10000
        
        if available_tokens <= min_chunk_tokens:
            raise ValueError(f"Размер чанка слишком мал. Требуется минимум {min_chunk_tokens + prompt_tokens + 2000} токенов")
        
        chunks = []
        current_chunk = ""
        
        # Разбиваем по абзацам
        paragraphs = text.split('\n\n')
        
        for paragraph in paragraphs:
            paragraph_tokens = self.count_tokens(paragraph)
            current_chunk_tokens = self.count_tokens(current_chunk)
            
            # Если добавление этого абзаца превысит лимит
            if current_chunk_tokens + paragraph_tokens > available_tokens:
                # Проверяем, достигли ли мы минимального размера
                if current_chunk_tokens >= min_chunk_tokens:
                    chunks.append(current_chunk.strip())
                    current_chunk = paragraph
                elif current_chunk:
                    # Если чанк меньше минимума, продолжаем добавлять
                    # Разбиваем большой абзац по предложениям
                    sentences = paragraph.split('. ')
                    for sentence in sentences:
                        sentence_with_dot = sentence + ". " if not sentence.endswith('.') else sentence + " "
                        
                        if self.count_tokens(current_chunk + sentence_with_dot) > available_tokens:
                            if self.count_tokens(current_chunk) >= min_chunk_tokens:
                                chunks.append(current_chunk.strip())
                                current_chunk = sentence_with_dot
                            else:
                                # Принудительно добавляем предложение, если чанк еще мал
                                current_chunk += sentence_with_dot
                        else:
                            current_chunk += sentence_with_dot
                else:
                    # Первый абзац слишком большой, разбиваем по предложениям
                    sentences = paragraph.split('. ')
                    for sentence in sentences:
                        sentence_with_dot = sentence + ". " if not sentence.endswith('.') else sentence + " "
                        
                        if self.count_tokens(current_chunk + sentence_with_dot) > available_tokens:
                            if current_chunk and self.count_tokens(current_chunk) >= min_chunk_tokens:
                                chunks.append(current_chunk.strip())
                                current_chunk = sentence_with_dot
                            else:
                                # Разбиваем по словам, если предложение слишком большое
                                words = sentence.split()
                                for word in words:
                                    word_with_space = word + " "
                                    if self.count_tokens(current_chunk + word_with_space) > available_tokens:
                                        if current_chunk and self.count_tokens(current_chunk) >= min_chunk_tokens:
                                            chunks.append(current_chunk.strip())
                                            current_chunk = word_with_space
                                        else:
                                            current_chunk += word_with_space
                                    else:
                                        current_chunk += word_with_space
                        else:
                            current_chunk += sentence_with_dot
            else:
                current_chunk += paragraph + "\n\n"
        
        # Добавляем последний чанк
        if current_chunk:
            current_chunk_tokens = self.count_tokens(current_chunk)
            if current_chunk_tokens >= min_chunk_tokens or not chunks:
                chunks.append(current_chunk.strip())
            elif chunks:
                # Если последний чанк слишком мал, добавляем к предыдущему
                chunks[-1] += "\n\n" + current_chunk.strip()
        
        # Проверяем, что все чанки соответствуют минимальному размеру
        final_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_tokens = self.count_tokens(chunk)
            if chunk_tokens < min_chunk_tokens and i < len(chunks) - 1:
                # Объединяем с следующим чанком
                chunks[i + 1] = chunk + "\n\n" + chunks[i + 1]
            else:
                final_chunks.append(chunk)
                
        print(f"📏 Минимальный размер чанка: {min_chunk_tokens:,} токенов")
        for i, chunk in enumerate(final_chunks):
            chunk_tokens = self.count_tokens(chunk)
            print(f"   Чанк {i + 1}: {chunk_tokens:,} токенов")
            
        return final_chunks
    
    async def process_chunk_with_retry(self, session: aiohttp.ClientSession, chunk: str, prompt: str, chunk_index: int, max_retries: int = 3) -> dict:
        """Обработка чанка с повторными попытками при rate limit"""
        for attempt in range(max_retries):
            try:
                result = await self.process_chunk(session, chunk, prompt, chunk_index)
                if result["success"]:
                    return result
                    
                # Если ошибка 429 (rate limit), ждем и повторяем
                if "429" in str(result.get("error", "")):
                    if self.paid_tier:
                        wait_time = 15 * (attempt + 1)  # Меньше ожидания для платного
                    else:
                        wait_time = 60 * (attempt + 1)  # Больше для free tier
                    
                    print(f"⏳ Rate limit для чанка {chunk_index + 1}, ожидание {wait_time}с (попытка {attempt + 1}/{max_retries})")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    return result
                    
            except Exception as e:
                print(f"❌ Ошибка в попытке {attempt + 1} для чанка {chunk_index + 1}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(5 * (attempt + 1))
                    
        return {
            "chunk_index": chunk_index,
            "success": False,
            "error": f"Не удалось обработать после {max_retries} попыток"
        }
    
    async def process_chunk(self, session: aiohttp.ClientSession, chunk: str, prompt: str, chunk_index: int) -> dict:
        """Обработка одного чанка через Gemini API"""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        
        headers = {"Content-Type": "application/json"}
        
        data = {
            "contents": [{"parts": [{"text": f"{prompt}\n\nТекст для обработки:\n{chunk}"}]}],
            "generationConfig": {
                "temperature": 0.7,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 8192 if self.paid_tier else 4096,
                "candidateCount": 1
            },
            "safetySettings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
            ]
        }
        
        try:
            timeout = aiohttp.ClientTimeout(total=300 if self.paid_tier else 180)
            async with session.post(url, headers=headers, json=data, timeout=timeout) as response:
                if response.status == 200:
                    result = await response.json()
                    
                    if "candidates" in result and len(result["candidates"]) > 0:
                        candidate = result["candidates"][0]
                        if "content" in candidate and "parts" in candidate["content"]:
                            content = candidate["content"]["parts"][0]["text"]
                            
                            input_tokens = self.count_tokens(chunk + prompt)
                            output_tokens = self.count_tokens(content)
                            total_tokens = input_tokens + output_tokens
                            
                            print(f"✅ Чанк {chunk_index + 1}: {input_tokens:,} + {output_tokens:,} = {total_tokens:,} токенов")
                            return {
                                "chunk_index": chunk_index,
                                "success": True,
                                "content": content,
                                "tokens_used": total_tokens,
                                "input_tokens": input_tokens,
                                "output_tokens": output_tokens
                            }
                        else:
                            return {"chunk_index": chunk_index, "success": False, "error": "Пустой ответ от Gemini"}
                    else:
                        return {"chunk_index": chunk_index, "success": False, "error": "Нет кандидатов в ответе"}
                else:
                    error_text = await response.text()
                    return {"chunk_index": chunk_index, "success": False, "error": f"HTTP {response.status}: {error_text}"}
                    
        except Exception as e:
            return {"chunk_index": chunk_index, "success": False, "error": str(e)}
    
    async def process_file(self, file_path: str, prompt: str, output_path: str, chunk_size: int = 1000000, delay: int = 0, concurrent: int = 1):
        """Основная функция обработки файла с полной параллелизацией"""
        print(f"📂 Загрузка файла: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"❌ Ошибка чтения файла: {e}")
            return False
        
        total_tokens = self.count_tokens(content)
        file_size_mb = len(content) / 1024 / 1024
        
        print(f"📏 Размер файла: {len(content):,} символов ({file_size_mb:.1f}MB)")
        print(f"🔢 Примерное количество токенов: {total_tokens:,}")
        
        chunks = self.split_into_chunks(content, prompt, chunk_size)
        print(f"📦 Файл разбит на {len(chunks)} чанков (мин. 10,000 токенов каждый)")
        
        # Рассчитываем время обработки для параллельной обработки
        estimated_time_per_chunk = 15 if not self.paid_tier else 8
        estimated_time = (len(chunks) * estimated_time_per_chunk) / concurrent + (len(chunks) * delay / concurrent)
        estimated_minutes = estimated_time / 60
        
        print(f"⏱️ Примерное время обработки: {estimated_minutes:.1f} минут")
        print(f"🚀 Полная параллельная обработка: {concurrent} одновременных запросов")
        
        # Создаем семафор для ограничения количества одновременных запросов
        semaphore = asyncio.Semaphore(concurrent)
        
        connector = aiohttp.TCPConnector(
            limit=concurrent * 3,  # Увеличиваем лимит соединений
            limit_per_host=concurrent * 2,
            ttl_dns_cache=300,
            use_dns_cache=True,
        )
        timeout = aiohttp.ClientTimeout(total=600)
        
        async def process_with_semaphore(session, chunk, i):
            async with semaphore:
                start_time = asyncio.get_event_loop().time()
                print(f"🔄 Начинаем чанк {i + 1}/{len(chunks)} ({datetime.now().strftime('%H:%M:%S')})")
                
                result = await self.process_chunk_with_retry(session, chunk, prompt, i)
                
                end_time = asyncio.get_event_loop().time()
                duration = end_time - start_time
                
                if result.get("success"):
                    print(f"✅ Завершен чанк {i + 1}/{len(chunks)} за {duration:.1f}с")
                else:
                    print(f"❌ Ошибка в чанке {i + 1}/{len(chunks)}: {result.get('error', 'Неизвестная ошибка')}")
                
                # Применяем задержку только если она установлена
                if delay > 0:
                    print(f"⏸️ Задержка {delay}с после чанка {i + 1}...")
                    await asyncio.sleep(delay)
                
                return result
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            # Всегда используем параллельную обработку, даже при concurrent=1
            print(f"🚀 Запускаем {len(chunks)} задач параллельно с ограничением {concurrent} одновременных")
            
            start_time = time.time()
            
            # Создаем все задачи сразу
            tasks = [process_with_semaphore(session, chunk, i) for i, chunk in enumerate(chunks)]
            
            # Запускаем все задачи параллельно
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            end_time = time.time()
            total_duration = end_time - start_time
            
            print(f"\n⏱️ Время выполнения всех задач: {total_duration / 60:.1f} минут")
            
            # Обрабатываем исключения
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    results[i] = {
                        "chunk_index": i,
                        "success": False,
                        "error": str(result)
                    }
        
        # Сохранение результатов
        successful_results = [r for r in results if isinstance(r, dict) and r.get("success")]
        failed_results = [r for r in results if isinstance(r, dict) and not r.get("success")]
        
        if failed_results:
            print(f"\n⚠️ Не удалось обработать {len(failed_results)} чанков:")
            for failed in failed_results[:5]:  # Показываем только первые 5 ошибок
                print(f"   Чанк {failed.get('chunk_index', '?') + 1}: {failed.get('error', 'Неизвестная ошибка')}")
            if len(failed_results) > 5:
                print(f"   ... и еще {len(failed_results) - 5} ошибок")
        
        if successful_results:
            try:
                # Сортируем результаты по индексу чанка
                successful_results.sort(key=lambda x: x['chunk_index'])
                
                final_content = "\n\n".join([
                    f"=== РЕЗУЛЬТАТ ЧАНКА {r['chunk_index'] + 1} ===\n{r['content']}"
                    for r in successful_results
                ])
                
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(final_content)
                
                total_tokens = sum(r.get("tokens_used", 0) for r in successful_results)
                total_input_tokens = sum(r.get("input_tokens", 0) for r in successful_results)
                total_output_tokens = sum(r.get("output_tokens", 0) for r in successful_results)
                
                print(f"\n✅ Обработка завершена!")
                print(f"   📊 Успешно: {len(successful_results)}/{len(chunks)} чанков")
                print(f"   🔢 Всего токенов: {total_tokens:,}")
                print(f"   📥 Входящие: {total_input_tokens:,}")
                print(f"   📤 Исходящие: {total_output_tokens:,}")
                print(f"   💾 Результат: {output_path}")
                print(f"   ⚡ Эффективность параллелизации: {(len(chunks) * estimated_time_per_chunk) / total_duration:.1f}x")
                
                # Показываем размер результата
                result_size = len(final_content) / 1024
                print(f"   📏 Размер результата: {result_size:.1f}KB")
                
                return len(failed_results) == 0  # True если все чанки успешны
                
            except Exception as e:
                print(f"❌ Ошибка сохранения результата: {e}")
                return False
        else:
            print("\n❌ Ни один чанк не был успешно обработан")
            return False

def main():
    parser = argparse.ArgumentParser(description="Обработка больших файлов через Gemini API")
    parser.add_argument("--file", required=True, help="Путь к входному файлу")
    parser.add_argument("--prompt", required=True, help="Промпт для обработки")
    parser.add_argument("--output", required=True, help="Путь к выходному файлу")
    parser.add_argument("--model", default="gemini-2.0-flash", help="Модель Gemini")
    parser.add_argument("--chunk-size", type=int, default=1000000, help="Размер чанка в токенах (мин. 12000)")
    parser.add_argument("--delay", type=int, default=0, help="Задержка между запросами в секундах")
    parser.add_argument("--concurrent", type=int, default=1, help="Количество параллельных запросов")
    parser.add_argument("--paid-tier", action="store_true", help="Использовать лимиты платного уровня")
    parser.add_argument("--api-key", help="API ключ Gemini")
    
    args = parser.parse_args()
    
    api_key = args.api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ Не указан API ключ Gemini")
        print("Установите переменную GEMINI_API_KEY или используйте --api-key")
        sys.exit(1)
    
    if not os.path.exists(args.file):
        print(f"❌ Файл не найден: {args.file}")
        sys.exit(1)
    
    # Валидация параметров
    if args.concurrent < 1 or args.concurrent > 20:
        print("❌ Количество параллельных запросов должно быть от 1 до 20")
        sys.exit(1)
    
    # Новое ограничение с учетом минимального размера чанка
    if args.chunk_size < 12000 or args.chunk_size > 5000000:
        print("❌ Размер чанка должен быть от 12,000 до 5,000,000 токенов")
        print("   (минимальный размер 10,000 + резерв для промпта и ответа)")
        sys.exit(1)
    
    processor = GeminiFileProcessor(api_key, args.model, args.paid_tier)
    
    start_time = time.time()
    
    try:
        success = asyncio.run(processor.process_file(
            args.file, 
            args.prompt, 
            args.output,
            args.chunk_size,
            args.delay,
            args.concurrent
        ))
        
        end_time = time.time()
        duration = end_time - start_time
        duration_minutes = duration / 60
        
        print(f"\n🏁 Общее время выполнения: {duration_minutes:.1f} минут")
        
        if success:
            print("🎉 Все чанки обработаны успешно!")
            sys.exit(0)
        else:
            print("⚠️ Обработка завершена с ошибками")
            sys.exit(2)  # Частичный успех
            
    except KeyboardInterrupt:
        print("\n⚠️ Обработка прервана пользователем")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
