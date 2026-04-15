# UI側との接続
def check_video(file):
    return {
        "result": "NG",
        "details": [
            "該当箇所：〇秒",
            "NGワード：〇〇",
            "該当箇所：〇秒",
            "誤字の可能性：△△"
        ]
    }