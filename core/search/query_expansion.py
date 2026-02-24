import re
from typing import List, Set


RU_SYNONYMS = {
    "файл": ["документ", "данные", "file"],
    "файлы": ["документы", "данные", "files"],
    "код": ["исходник", "скрипт", "code"],
    "память": ["memory", "хранение", "запоминание"],
    "поиск": ["найти", "отыскать", "search"],
    "тест": ["тестирование", "проверка", "unit"],
    "ошибка": ["баг", "проблема", "error"],
    "функция": ["метод", "процедура", "function"],
    "класс": ["объект", "сущность", "class"],
    "данные": ["информация", "контент", "data"],
    "настройка": ["конфиг", "параметр", "setting"],
    "пользователь": ["юзер", "клиент", "user"],
    "сервер": ["бэкенд", "служба", "server"],
    "запрос": ["request", "вызов"],
    "ответ": ["response", "результат"],
}

UK_SYNONYMS = {
    "файл": ["документ", "дані", "file"],
    "файли": ["документи", "дані", "files"],
    "код": ["скрипт", "програма", "code"],
    "память": ["memory", "зберігання", "запам'ятовування"],
    "пошук": ["знайти", "відшукати", "search"],
    "тест": ["тестування", "перевірка", "unit"],
    "помилка": ["баг", "проблема", "error"],
    "функція": ["метод", "процедура", "function"],
    "клас": ["об'єкт", "сутність", "class"],
    "дані": ["інформація", "контент", "data"],
    "налаштування": ["конфіг", "параметр", "setting"],
    "користувач": ["клієнт", "user", "account"],
    "сервер": ["бекенд", "служба", "server"],
    "запит": ["request", "виклик"],
    "відповідь": ["response", "результат"],
}

EN_SYNONYMS = {
    "file": ["doc", "document", "data"],
    "code": ["source", "script", "program"],
    "memory": ["storage", "save"],
    "search": ["find", "lookup", "query"],
    "test": ["testing", "check", "unit"],
    "error": ["bug", "issue", "problem"],
    "function": ["method", "procedure"],
    "class": ["object", "entity"],
    "data": ["info", "information"],
    "config": ["setting", "option", "parameter"],
    "user": ["client", "enduser"],
    "server": ["backend", "service"],
    "client": ["frontend"],
    "request": ["call", "invoke"],
    "response": ["result", "return"],
}


COMMON_EXTENSIONS = {
    "py": "python",
    "js": "javascript",
    "ts": "typescript",
    "json": "json",
    "md": "markdown",
    "txt": "text",
    "html": "html",
    "css": "css",
    "sql": "sql",
    "yml": "yaml",
    "yaml": "yaml",
    "xml": "xml",
    "sh": "shell",
    "bash": "shell",
    "zsh": "shell",
    "go": "go",
    "rs": "rust",
    "java": "java",
    "c": "c",
    "cpp": "cpp",
    "h": "header",
    "hpp": "cpp",
}


class QueryExpander:
    """Query expansion using synonyms and related terms."""

    def __init__(self):
        self.ru_synonyms = RU_SYNONYMS
        self.uk_synonyms = UK_SYNONYMS
        self.en_synonyms = EN_SYNONYMS
        self._all_synonyms = [self.ru_synonyms, self.uk_synonyms, self.en_synonyms]
        self.extensions = COMMON_EXTENSIONS

    def _tokenize(self, query: str) -> List[str]:
        """Tokenize query into words."""
        normalized = (query or "").lower().replace("'", "").replace("’", "")
        tokens = re.findall(r"[^\W_]{2,}", normalized, flags=re.UNICODE)
        return tokens

    def _get_synonyms(self, word: str) -> Set[str]:
        """Get synonyms for a word across RU/UK/EN dictionaries.

        For other languages, tokenization still works universally and this
        method simply returns no dictionary-based expansions.
        """
        synonyms = set()
        word_lower = word.lower()

        for synonym_map in self._all_synonyms:
            if word_lower in synonym_map:
                synonyms.update(synonym_map[word_lower])

            for syn_group in synonym_map.values():
                if word_lower in syn_group:
                    synonyms.add(word_lower)
                    synonyms.update(syn_group)

        return synonyms

    def _expand_file_extensions(self, query: str) -> Set[str]:
        """Add common file extensions."""
        expansions = set()

        for ext, lang in self.extensions.items():
            if ext in query.lower():
                expansions.add(lang)

        return expansions

    def expand(self, query: str, max_terms: int = 5) -> List[str]:
        """Expand query with synonyms and related terms."""
        if not query:
            return []

        tokens = self._tokenize(query)
        expansions: Set[str] = set()

        for token in tokens:
            synonyms = self._get_synonyms(token)
            expansions.update(synonyms)

        file_expansions = self._expand_file_extensions(query)
        expansions.update(file_expansions)

        all_terms = list(expansions)[:max_terms]

        return all_terms

    def expand_query(self, query: str, max_terms: int = 5) -> str:
        """Return expanded query string."""
        if not query:
            return query

        expansions = self.expand(query, max_terms)

        if not expansions:
            return query

        expanded = f"{query} {' '.join(expansions)}"
        return expanded


query_expander = QueryExpander()
