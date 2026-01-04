import sqlite3
import pandas as pd

# 数据库路径
db_path = '/data/nishome/user1/chaochuan/TSGym_benchmark/long_term_forecast_TSGym_transformer_log.db'

print(f"正在检查数据库: {db_path}")

try:
    conn = sqlite3.connect(db_path)
    
    # 1. 检查是否存在 exp_logs 表
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='exp_logs';")
    if not cursor.fetchone():
        print("错误: 数据库中未找到 'exp_logs' 表。")
        # 列出所有表
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"存在的表: {[t[0] for t in tables]}")
    else:
        # 2. 查询失败的实验 (status='FAILED' 或 error_msg 不为空)
        print("正在查询 2025-12-12 之后的失败实验记录...")
        query = """
        SELECT exp_setting, start_time, end_time, status, error_msg 
        FROM exp_logs 
        WHERE (status = 'FAILED' OR (error_msg IS NOT NULL AND error_msg != ''))
        AND (exp_setting LIKE '%ETTh1%' OR exp_setting LIKE '%ETTh2%')
        AND start_time > '2025-12-12'
        ORDER BY start_time DESC
        """
        
        df_errors = pd.read_sql_query(query, conn)
        
        count = len(df_errors)
        print(f"\n>>> 发现 {count} 条失败记录 <<<")
        
        if count > 0:
            # 设置 pandas 显示选项以便完整查看错误信息
            pd.set_option('display.max_colwidth', None)
            pd.set_option('display.width', 1000)
            
            # 3. 打印错误摘要 (按错误信息分组统计)
            print("\n=== 错误类型统计 (Top 10) ===")
            error_counts = df_errors['error_msg'].value_counts().head(10)
            print(error_counts)
            
            # 4. 打印详细记录 (最近的 5 条)
            print("\n=== 最近 5 条失败详情 ===")
            print(df_errors.head(5)[['start_time', 'exp_setting', 'error_msg']].to_string(index=False))
        else:
            print("恭喜！没有发现任何失败的实验记录。")

    conn.close()

except Exception as e:
    print(f"发生异常: {e}")