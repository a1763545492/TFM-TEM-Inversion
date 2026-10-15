# -*- coding: utf-8 -*-
"""
tfm_engine.project_db (M19)
=============================
SQLite 项目库 — 替代目录约定管理测线/结果。

数据库表结构:
  projects     : 项目元数据（名称、日期、坐标系、备注）
  lines        : 测线元数据（项目ID、线号、起始/结束坐标、Tx参数）
  stations     : 测站元数据（线ID、桩号、XYZ坐标、DAT文件路径）
  data_files   : 数据文件记录（DAT/HDR/Excel 路径、哈希、导入时间）
  inversion_runs : 反演运行记录（算法、参数、时间、RMSE、结果路径）
  qc_flags     : 质量控制标记（测站ID、标记类型、备注）

功能:
  - 项目创建/打开/关闭
  - 测线/测站批量导入（从 Excel/DAT）
  - 反演结果注册（自动记录参数和指标）
  - 查询: 按项目/线/站检索
  - 导出: 生成项目摘要报告（HTML/CSV）

完全离线，标准库 sqlite3。
"""
from __future__ import annotations
from typing import Optional, Callable, List, Dict, Any
from pathlib import Path
import sqlite3
import json
import hashlib
import datetime


# ============================================================
# 数据库模式
# ============================================================

SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    coordinate_system TEXT DEFAULT 'local',
    created_at TEXT DEFAULT (datetime('now')),
    metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS lines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL REFERENCES projects(id),
    line_number INTEGER NOT NULL,
    name TEXT,
    start_x REAL, start_y REAL,
    end_x REAL, end_y REAL,
    tx_area_m2 REAL,
    tx_current_a REAL,
    n_stations INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now')),
    UNIQUE(project_id, line_number)
);

CREATE TABLE IF NOT EXISTS stations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    line_id INTEGER NOT NULL REFERENCES lines(id),
    station_number INTEGER NOT NULL,
    code TEXT,
    x REAL, y REAL, z REAL,
    dat_file TEXT,
    hdr_file TEXT,
    excel_row INTEGER,
    qc_flag TEXT DEFAULT 'good',
    created_at TEXT DEFAULT (datetime('now')),
    UNIQUE(line_id, station_number)
);

CREATE TABLE IF NOT EXISTS data_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    station_id INTEGER REFERENCES stations(id),
    file_path TEXT NOT NULL,
    file_type TEXT,  -- 'dat' / 'hdr' / 'excel' / 'result'
    file_hash TEXT,
    file_size INTEGER,
    imported_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS inversion_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL REFERENCES projects(id),
    line_id INTEGER REFERENCES lines(id),
    algorithm TEXT NOT NULL,
    parameters_json TEXT,
    n_stations INTEGER,
    rmse_mean REAL,
    rmse_std REAL,
    result_path TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS qc_flags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    station_id INTEGER NOT NULL REFERENCES stations(id),
    flag_type TEXT NOT NULL,  -- 'static' / 'noise' / 'outlier' / 'manual'
    severity TEXT DEFAULT 'warning',  -- 'info' / 'warning' / 'error'
    comment TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_lines_project ON lines(project_id);
CREATE INDEX IF NOT EXISTS idx_stations_line ON stations(line_id);
CREATE INDEX IF NOT EXISTS idx_stations_code ON stations(code);
CREATE INDEX IF NOT EXISTS idx_inversion_project ON inversion_runs(project_id);
CREATE INDEX IF NOT EXISTS idx_qc_station ON qc_flags(station_id);
"""


# ============================================================
# 数据库管理类
# ============================================================

class ProjectDB:
    """SQLite 项目库管理器。"""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.conn: Optional[sqlite3.Connection] = None
        self._connect()

    def _connect(self):
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # --------------------------------------------------------
    # 项目操作
    # --------------------------------------------------------

    def create_project(self, name: str, description: str = '',
                       coordinate_system: str = 'local',
                       metadata: Optional[dict] = None) -> int:
        """创建项目，返回 project_id。"""
        meta_json = json.dumps(metadata or {}, ensure_ascii=False)
        cur = self.conn.execute(
            'INSERT INTO projects (name, description, coordinate_system, metadata_json) VALUES (?, ?, ?, ?)',
            (name, description, coordinate_system, meta_json)
        )
        self.conn.commit()
        return cur.lastrowid

    def get_project(self, name: str) -> Optional[dict]:
        """按名称获取项目。"""
        row = self.conn.execute('SELECT * FROM projects WHERE name = ?', (name,)).fetchone()
        return dict(row) if row else None

    def list_projects(self) -> List[dict]:
        """列出所有项目。"""
        rows = self.conn.execute('SELECT * FROM projects ORDER BY created_at DESC').fetchall()
        return [dict(r) for r in rows]

    # --------------------------------------------------------
    # 测线操作
    # --------------------------------------------------------

    def add_line(self, project_id: int, line_number: int, name: str = '',
                 start_x: float = 0, start_y: float = 0,
                 end_x: float = 0, end_y: float = 0,
                 tx_area_m2: float = 1e6, tx_current_a: float = 30.0) -> int:
        """添加测线，返回 line_id。"""
        cur = self.conn.execute(
            '''INSERT INTO lines (project_id, line_number, name, start_x, start_y, end_x, end_y, tx_area_m2, tx_current_a)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (project_id, line_number, name, start_x, start_y, end_x, end_y, tx_area_m2, tx_current_a)
        )
        self.conn.commit()
        return cur.lastrowid

    def get_line(self, project_id: int, line_number: int) -> Optional[dict]:
        row = self.conn.execute(
            'SELECT * FROM lines WHERE project_id = ? AND line_number = ?',
            (project_id, line_number)
        ).fetchone()
        return dict(row) if row else None

    def list_lines(self, project_id: int) -> List[dict]:
        rows = self.conn.execute(
            'SELECT * FROM lines WHERE project_id = ? ORDER BY line_number',
            (project_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    # --------------------------------------------------------
    # 测站操作
    # --------------------------------------------------------

    def add_station(self, line_id: int, station_number: int, code: str = '',
                    x: float = 0, y: float = 0, z: float = 0,
                    dat_file: str = '', hdr_file: str = '',
                    qc_flag: str = 'good') -> int:
        """添加测站，返回 station_id。"""
        cur = self.conn.execute(
            '''INSERT INTO stations (line_id, station_number, code, x, y, z, dat_file, hdr_file, qc_flag)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (line_id, station_number, code, x, y, z, dat_file, hdr_file, qc_flag)
        )
        self.conn.commit()
        return cur.lastrowid

    def add_stations_batch(self, line_id: int, stations: List[dict]):
        """批量添加测站。"""
        for st in stations:
            self.add_station(
                line_id=line_id,
                station_number=st['station_number'],
                code=st.get('code', ''),
                x=st.get('x', 0), y=st.get('y', 0), z=st.get('z', 0),
                dat_file=st.get('dat_file', ''),
                hdr_file=st.get('hdr_file', ''),
                qc_flag=st.get('qc_flag', 'good')
            )

    def list_stations(self, line_id: int, qc_filter: Optional[str] = None) -> List[dict]:
        """列出测站。qc_filter: 'good' / 'bad' / None(全部)"""
        if qc_filter:
            rows = self.conn.execute(
                'SELECT * FROM stations WHERE line_id = ? AND qc_flag = ? ORDER BY station_number',
                (line_id, qc_filter)
            ).fetchall()
        else:
            rows = self.conn.execute(
                'SELECT * FROM stations WHERE line_id = ? ORDER BY station_number',
                (line_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def find_station_by_code(self, code: str) -> Optional[dict]:
        row = self.conn.execute('SELECT * FROM stations WHERE code = ?', (code,)).fetchone()
        return dict(row) if row else None

    # --------------------------------------------------------
    # 反演运行记录
    # --------------------------------------------------------

    def register_inversion(self, project_id: int, line_id: int,
                           algorithm: str, parameters: dict,
                           n_stations: int, rmse_mean: float, rmse_std: float,
                           result_path: str) -> int:
        """注册反演运行，返回 run_id。"""
        cur = self.conn.execute(
            '''INSERT INTO inversion_runs (project_id, line_id, algorithm, parameters_json, n_stations, rmse_mean, rmse_std, result_path)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
            (project_id, line_id, algorithm, json.dumps(parameters, ensure_ascii=False),
             n_stations, rmse_mean, rmse_std, result_path)
        )
        self.conn.commit()
        return cur.lastrowid

    def list_inversions(self, project_id: int, line_id: Optional[int] = None) -> List[dict]:
        """查询反演记录。"""
        if line_id:
            rows = self.conn.execute(
                'SELECT * FROM inversion_runs WHERE project_id = ? AND line_id = ? ORDER BY created_at DESC',
                (project_id, line_id)
            ).fetchall()
        else:
            rows = self.conn.execute(
                'SELECT * FROM inversion_runs WHERE project_id = ? ORDER BY created_at DESC',
                (project_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    # --------------------------------------------------------
    # QC 标记
    # --------------------------------------------------------

    def add_qc_flag(self, station_id: int, flag_type: str,
                    severity: str = 'warning', comment: str = '') -> int:
        cur = self.conn.execute(
            'INSERT INTO qc_flags (station_id, flag_type, severity, comment) VALUES (?, ?, ?, ?)',
            (station_id, flag_type, severity, comment)
        )
        self.conn.commit()
        return cur.lastrowid

    def list_qc_flags(self, station_id: Optional[int] = None) -> List[dict]:
        if station_id:
            rows = self.conn.execute(
                'SELECT * FROM qc_flags WHERE station_id = ? ORDER BY created_at DESC',
                (station_id,)
            ).fetchall()
        else:
            rows = self.conn.execute('SELECT * FROM qc_flags ORDER BY created_at DESC').fetchall()
        return [dict(r) for r in rows]

    # --------------------------------------------------------
    # 统计与报告
    # --------------------------------------------------------

    def get_project_summary(self, project_id: int) -> dict:
        """获取项目摘要统计。"""
        lines = self.list_lines(project_id)
        total_stations = 0
        for line in lines:
            line_id = line['id']
            stations = self.list_stations(line_id)
            total_stations += len(stations)
            line['stations'] = stations

        inversions = self.list_inversions(project_id)
        qc_flags = self.list_qc_flags()

        return {
            'project_id': project_id,
            'n_lines': len(lines),
            'total_stations': total_stations,
            'n_inversions': len(inversions),
            'n_qc_flags': len(qc_flags),
            'lines': lines,
            'inversions': inversions[:10],  # 最近10条
        }

    def export_summary_html(self, project_id: int, out_html: str):
        """导出项目摘要 HTML。"""
        summary = self.get_project_summary(project_id)

        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>项目摘要 - {project_id}</title>
<style>
body {{ font-family: 'Microsoft YaHei', sans-serif; margin: 20px; }}
table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
th {{ background-color: #4CAF50; color: white; }}
h1, h2 {{ color: #333; }}
</style></head><body>
<h1>项目摘要报告</h1>
<p>项目ID: {project_id} | 测线数: {summary['n_lines']} | 测站总数: {summary['total_stations']} | 反演次数: {summary['n_inversions']}</p>

<h2>测线列表</h2>
<table><tr><th>线号</th><th>名称</th><th>测站数</th><th>Tx面积</th><th>Tx电流</th></tr>
"""
        for line in summary['lines']:
            html += f"<tr><td>{line['line_number']}</td><td>{line['name']}</td><td>{len(line['stations'])}</td><td>{line['tx_area_m2']:.0f}</td><td>{line['tx_current_a']:.1f}</td></tr>\n"

        html += """</table>

<h2>最近反演记录</h2>
<table><tr><th>算法</th><th>测站数</th><th>RMSE均值</th><th>RMSE标准差</th><th>时间</th></tr>
"""
        for inv in summary['inversions']:
            html += f"<tr><td>{inv['algorithm']}</td><td>{inv['n_stations']}</td><td>{inv['rmse_mean']:.3f}</td><td>{inv['rmse_std']:.3f}</td><td>{inv['created_at']}</td></tr>\n"

        html += """</table>
</body></html>"""

        Path(out_html).write_text(html, encoding='utf-8')


# ============================================================
# 便捷函数
# ============================================================

def create_project_db(db_path: str | Path, project_name: str) -> ProjectDB:
    """创建新项目库。"""
    db = ProjectDB(db_path)
    db.create_project(project_name)
    return db


def import_from_excel(db: ProjectDB, project_id: int,
                      excel_path: str | Path,
                      coord_loader_fn: Optional[Callable] = None,
                      log_fn: Optional[Callable] = None) -> dict:
    """从 Excel 元数据批量导入测线/测站。

    Parameters
    ----------
    db : ProjectDB
    project_id : int
    excel_path : str, Excel 文件路径
    coord_loader_fn : callable, 可选，用于加载坐标
    log_fn : callable

    Returns
    -------
    stats : dict, 导入统计
    """
    from .excel_meta import load_excel_meta

    meta = load_excel_meta(Path(excel_path))
    stats = {'lines_added': 0, 'stations_added': 0}

    for line_info in meta.get('lines', []):
        line_id = db.add_line(
            project_id=project_id,
            line_number=line_info['line_number'],
            name=line_info.get('name', ''),
            tx_area_m2=line_info.get('tx_area_m2', 1e6),
            tx_current_a=line_info.get('tx_current_a', 30.0)
        )
        stats['lines_added'] += 1

        stations = []
        for st in line_info.get('stations', []):
            stations.append({
                'station_number': st['station_number'],
                'code': st.get('code', ''),
                'x': st.get('x', 0), 'y': st.get('y', 0), 'z': st.get('z', 0),
                'dat_file': st.get('dat_file', ''),
            })
        db.add_stations_batch(line_id, stations)
        stats['stations_added'] += len(stations)

    if log_fn:
        log_fn(f"  [project_db] 导入完成: {stats['lines_added']} 线, {stats['stations_added']} 站")

    return stats


__all__ = [
    'ProjectDB',
    'create_project_db',
    'import_from_excel',
]
