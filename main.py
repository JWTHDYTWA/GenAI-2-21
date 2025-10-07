import argparse
from transformers import pipeline, TextGenerationPipeline


# Привязка к конкретной модели вызвана отличающимся форматом данных у разных моделей (проверено).
# При переключении модели может сломаться индексация контейнеров, из-за чего потребуется переписывать код.
# Примечание: Qwen3, в отличии от Gemma, не является Gated моделью и не требует токена HuggingFace
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
MAX_OUTPUT_TOKENS = 2048
# У модели Qwen3 ну очень уж внушительное
# для локальной модели контекстное окно в 262,144 токенов
MAX_INPUT_TOKENS = 32768
SYSTEM_PROMPT = "Твоя задача - генерировать новый текст в стиле, \
полностью соответствующем стилю образца, передаваемого пользователем. \
Тема не обязана совпадать с темой образца, важно лишь сохранить стиль."


def text_pipeline_init(lm: str) -> TextGenerationPipeline:
    """
    Создаёт экземпляр пайплайна для генерации текста с заданной моделью
    и автоматической конфигурацией для наилучшей производительности.

    Args:
        lm (str): Имя или путь к языковой модели.

    Returns:
        TextGenerationPipeline: Инициализированный пайплайн для генерации текста.
    """
    pipe = pipeline(
        task="text-generation",
        model=lm,
        # Для автоматического подбора следующих параметров используется библиотека `accelerate`
        device_map="auto",
        dtype="auto"
    )
    return pipe


def format_message(example: str):
    """
    Возвращает сообщение для языковой модели в формате ChatML.

    Args:
        example (str): Образец текста.

    Returns:
        list[dict[str,str]]
    """
    return [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': f'Образец:\n"{example}"'}
    ]


def inference(
        example: str,
        pipe: TextGenerationPipeline,
        out_token_limit: int | None = None,
        in_token_limit: int | None = None) -> str:
    """
    Генерирует новый текст на основе стиля из образца текста `example` с использованием заданного пайплайна.

    Args:
        example (str): Образец текста, стиль которого берётся за основу.
        pipe (TextGenerationPipeline): Инициализированный пайплайн TextGenerationPipeline из библиотеки transformers.
        out_token_limit (int | None): Ограничение на количество выходных токенов модели.
        in_token_limit (int | None): Ограничение на количество входных токенов.

    Returns:
        str: Текст, сгенерированный на основе образца.

    Raises:
        TypeError: Если один из параметров не соответствует ожидаемому типу.
        ValueError: Если функции передан пустой запрос.
    """

    message = None
    answer = None

    if not isinstance(example, str):
        raise TypeError('Аргумент example должен иметь тип str.')
    if not isinstance(pipe, TextGenerationPipeline):
        raise TypeError('Аргумент pipe должен иметь тип TextGenerationPipeline.')

    example = example.strip()
    if example == '':
        raise ValueError('Входная строка не должна быть пустой или состоять только из пробелов.')

    message = format_message(example)
    if in_token_limit is not None:
        # Токенизация - незатратная операция, поэтому
        # токенизируем сообщение и проверяем на превышение.
        assert pipe.tokenizer is not None, 'pipe.tokenizer не должен быть None'
        tokens = pipe.tokenizer.apply_chat_template(message, tokenize=True, add_generation_prompt=False)
        if len(tokens) > in_token_limit:
            raise ValueError('Количество входных токенов превышает установленное ограничение.')
    answer = pipe(message, max_new_tokens=out_token_limit, return_full_text=False)[0]['generated_text']

    assert isinstance(answer, str)
    return answer


def main():

    # Используется библиотека argparse для обработки параметров командной строки
    # Описание параметров командной строки можно получить, вызвав программу с флагом -h:
    # ./main.py -h
    parser = argparse.ArgumentParser('GenAI-1-21')
    parser.add_argument('input_file', nargs='?', default='input.txt', help='Путь к входному файлу. По умолчанию - "input.txt".')
    parser.add_argument('-o', '--output', default='output.txt', help='Путь к выходному файлу. По умолчанию - "output.txt".')
    parser.add_argument('-t', '--tokens', default=MAX_OUTPUT_TOKENS, type=int, help=f'Лимит output-токенов на один запрос. По умолчанию - {MAX_OUTPUT_TOKENS}.')
    args = parser.parse_args()

    # Инициализация пайплайна модели
    try:
        pipe_instance = text_pipeline_init(MODEL_NAME)
        print('Модель успешно инициализирована.')
    except Exception as e:
        print('\033[31m' + f'Произошла ошибка при инициализации модели:\n{e}' + '\033[0m')
        exit(1)
    
    # Обработка файла
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            text = f.read()

        # Передача входных данных модели и получение ответа
        answer = inference(text, pipe_instance, args.tokens, MAX_INPUT_TOKENS)

        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(answer)
        
    except FileNotFoundError as e:
        print('\033[31m' + f'Ошибка открытия файла:\n{e}' + '\033[0m')
    except Exception as e:
        print('\033[31m' + f'Ошибка обработки файла:\n{e}' + '\033[0m')


if __name__ == '__main__':
    main()
