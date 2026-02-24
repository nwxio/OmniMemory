import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..db import db, dumps, fetch_all


class AuditLogger:
    """Audit logging for memory operations."""
    
    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()
    
    async def log(
        self,
        action: str,
        resource_type: str,
        resource_id: str = "",
        user_id: str = "",
        details: Optional[Dict[str, Any]] = None,
        ip_address: str = "",
    ) -> None:
        """Log an audit event."""
        now = self._utc_now()
        
        async with db.connect() as conn:
            await conn.execute(
                """INSERT INTO audit_log 
                   (timestamp, user_id, action, resource_type, resource_id, details, ip_address)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (now, user_id, action, resource_type, resource_id, dumps(details or {}), ip_address),
            )
            await conn.commit()
    
    async def get_logs(
        self,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        user_id: Optional[str] = None,
        action: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get audit logs with filters."""
        conditions: List[str] = []
        params: List[Any] = []
        
        if resource_type:
            conditions.append("resource_type = ?")
            params.append(resource_type)
        if resource_id:
            conditions.append("resource_id = ?")
            params.append(resource_id)
        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)
        if action:
            conditions.append("action = ?")
            params.append(action)
        
        where = " AND ".join(conditions) if conditions else "1=1"
        params.append(limit)
        
        async with db.connect() as conn:
            rows = await fetch_all(
                conn,
                f"""SELECT timestamp, user_id, action, resource_type, resource_id, details, ip_address 
                    FROM audit_log WHERE {where} ORDER BY timestamp DESC LIMIT ?""",
                params,
            )
        
        results = []
        for row in rows:
            results.append({
                "timestamp": row["timestamp"],
                "user_id": row["user_id"],
                "action": row["action"],
                "resource_type": row["resource_type"],
                "resource_id": row["resource_id"],
                "details": json.loads(row["details"] or "{}"),
                "ip_address": row["ip_address"],
            })
        return results
    
    async def get_user_activity(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get activity for a specific user."""
        return await self.get_logs(user_id=user_id, limit=limit)
    
    async def get_resource_history(self, resource_type: str, resource_id: str) -> List[Dict[str, Any]]:
        """Get history for a specific resource."""
        return await self.get_logs(resource_type=resource_type, resource_id=resource_id)


audit_logger = AuditLogger()
