import streamlit as st
import time
import sys
import os
import json

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.checker import check_video

st.title("動画チェック自動化ツール")
st.write("動画内のテロップを解析し、誤字やNGワードを自動検出します")

uploaded_file = st.file_uploader("動画をアップロード")

if uploaded_file:
    file_name = uploaded_file.name.lower()

    if not (file_name.endswith(".mp4") or file_name.endswith(".mov")):
        st.error("対応していないファイル形式です（mp4 / mov のみ対応）")
        st.stop()

    else:
        st.success("アップロード完了")

        if st.button("チェック実行"):
            status = st.empty()
            status.write("チェック準備中...")

            progress = st.progress(0)

            for i in range(100):
                time.sleep(0.01)
                progress.progress(i + 1)

                if i == 30:
                    status.write("音声解析中...")
                elif i == 60:
                    status.write("テロップ解析中...")
                elif i == 90:
                    status.write("最終チェック中...")

            status.write("チェック完了！")

            result = check_video(uploaded_file)
            res = result["result"]

            if res == "OK":
                st.success(f"結果：{res}")
            elif res == "注意":
                st.warning(f"結果：{res}")
            elif res == "NG":
                st.error(f"結果：{res}")

            st.write("■ 詳細")
            for d in result["details"]:
                st.write(f"- {d}")

            # ダウンロード用データ作成
            download_data = {
                "filename": uploaded_file.name,
                "result": result["result"],
                "details": result["details"]
            }

            json_str = json.dumps(download_data, ensure_ascii=False, indent=2)

            st.download_button(
                label="結果をダウンロード",
                data=json_str,
                file_name="check_result.json",
                mime="application/json"
            )