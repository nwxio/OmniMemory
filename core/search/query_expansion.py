import re
from typing import List, Set


RU_SYNONYMS = {
    "файл": ["файлик", "документ", "данные"],
    "файлы": ["файлики", "документы", "данные"],
    "код": ["исходник", "программа", "скрипт"],
    "память": ["memory", "сохранение", "запоминание"],
    "поиск": ["найти", "отыскать", "search"],
    "тест": ["тестирование", "проверка", "юнит"],
    "ошибка": ["баг", "проблема", "issue"],
    "функция": ["метод", "процедура", "функционал"],
    "класс": ["объект", "сущность"],
    "данные": ["информация", "сведения", "контент"],
    "настройка": ["конфиг", "параметр", "опция"],
    "пользователь": ["юзер", "user", "клиент"],
    "сервер": ["бэкенд", "backend", "служба"],
    "клиент": ["фронтенд", "frontend"],
    "запрос": ["request", "вызов"],
    "ответ": ["response", "результат"],
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
        self.en_synonyms = EN_SYNONYMS
        self.extensions = COMMON_EXTENSIONS
    
    def _tokenize(self, query: str) -> List[str]:
        """Tokenize query into words."""
        tokens = re.findall(r"[^\W_]{2,}", query.lower(), flags=re.UNICODE)
        return tokens
    
    def _get_synonyms(self, word: str) -> Set[str]:
        """Get synonyms for a word."""
        synonyms = set()
        word_lower = word.lower()
        
        if word_lower in self.ru_synonyms:
            synonyms.update(self.ru_synonyms[word_lower])
        
        if word_lower in self.en_synonyms:
            synonyms.update(self.en_synonyms[word_lower])
        
        for syn_group in self.ru_synonyms.values():
            if word_lower in syn_group:
                synonyms.add(word_lower)
                synonyms.update(syn_group)
        
        for syn_group in self.en_synonyms.values():
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
