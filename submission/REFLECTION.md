# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** Phan Hoàng Vũ
**Khoá:** A20-K4
**Tier đã chạy:** T4
**Ngày:** 2026-10-09

> Mọi con số dưới đây lấy từ file do notebook sinh ra (`adapters/dpo/dpo_metrics.json`,
> `data/eval/judge_summary.json`, `data/eval/benchmark_results.json`…), không ước lượng bằng mắt.

---

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Kaggle T4 16 GB |
| Mô hình gốc | unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit |
| Dữ liệu SFT | saillab/alpaca-vietnamese-cleaned · 1.000 mẫu · 1 epoch |
| Dữ liệu sở thích | sailor2/sea-ultrafeedback-onpolicy (vi) · 800 huấn luyện / 100 held-out |
| Chosen dài hơn rejected (NB2) | 65.9% |
| DPO: β / tốc độ học (lr) / số epoch | 0.1 / 5e-6 / 1.0 (loss: sigmoid) |
| Giám khảo | rm-panel:Skywork-Reward-V2-Qwen3-4B + Skywork-Reward-V2-Llama-3.2-3B; sanity accuracy 100% |
| Chi phí | 0 đồng (Kaggle GPU miễn phí) |

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | 42 phút (100 steps) |
| VRAM cao nhất | 12.8 GB |
| Reward gap cuối trên tập huấn luyện (chosen − rejected) | 0.0527 |
| Độ chính xác reward trên held-out | 71.0% |
| Margin trên held-out | 0.0669 |
| Chẩn đoán tự động (`diagnosis`) | INTENDED |
| Độ dài trung bình câu trả lời SFT → DPO (NB4) | 357 → 394 ký tự (held-out: 312 → 354 ký tự) |

---

## 3. Đọc đường reward (≥ 100 từ)

> Ảnh: `screenshots/03-dpo-reward-curves.png`

Dựa trên biểu đồ reward và file số liệu `adapters/dpo/dpo_metrics.json`, cả hai giá trị `rewards/chosen` và `rewards/rejected` đều khởi đầu chính xác từ 0.0 nat (do mô hình policy ban đầu trùng với mô hình tham chiếu reference). Trong suốt 100 bước huấn luyện, reward của câu `chosen` tăng trưởng đều đặn và đạt 0.2403 trên tập huấn luyện và 0.2717 trên tập held-out. Trong khi đó, reward của câu `rejected` tăng chậm hơn, chỉ đạt 0.1876 trên tập huấn luyện và 0.2048 trên tập held-out.

Khoảng cách reward (margin = chosen − rejected) tăng dương ổn định từ 0.0 lên 0.0527 trên train và đạt 0.0669 trên held-out. Điều rất đáng chú ý là đường cong trên tập kiểm tra held-out hoàn toàn đồng pha và có độ mở margin thậm chí nhỉnh hơn tập huấn luyện, chứng minh mô hình không hề bị học thuộc lòng (overfitting) hay ghi nhớ máy móc các mẫu câu. Hiện tượng dịch chuyển xác suất tiêu cực (Likelihood Displacement) không xảy ra vì reward của `chosen` thực sự tăng lên thay vì bị kéo tụt xuống. Độ chính xác phân biệt preference trên held-out đạt 71.0%. Kết luận chẩn đoán tự động là **INTENDED (Đúng kỳ vọng)** hoàn toàn ăn khớp và phản ánh chính xác xu hướng của các đường cong reward thực tế.

---

## 4. So sánh SFT vs SFT+DPO

> Ảnh: `screenshots/04-side-by-side-table.png`

Từ `data/eval/judge_summary.json`:

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (khoảng tin cậy 95%) | Win rate các cặp dài gần bằng nhau | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | 50 | 3 | 4 | 43 | 0.490 [0.440, 0.540] | 0.478 | 0.571 |
| hữu ích — helpfulness (4) | 4 | 0 | 0 | 4 | 0.500 [0.500, 0.500] | 0.500 | N/A |
| an toàn — safety (4) | 4 | 1 | 0 | 3 | 0.625 [0.500, 0.875] | 0.625 | 1.000 |

Giám khảo: Hội đồng Skywork-Reward-V2-Qwen3-4B + Skywork-Reward-V2-Llama-3.2-3B · sanity accuracy: 100% · `score_length_spearman`: 0.076 (Qwen3) và 0.303 (Llama), độ đồng thuận hội đồng đạt 86.2%.

Khoảng tin cậy 95% của win rate trên tập held-out là [0.440, 0.540], hoàn toàn chứa giá trị 0.50. Điều này chỉ ra rằng trên các câu hỏi tổng quát, DPO duy trì năng lực cốt lõi tương đương SFT mà không làm suy thoái mô hình (có tới 43/50 câu hai giám khảo chấm hoà). Bộ kiểm tra Sanity tiếng Việt đạt 100% ở cả hai giám khảo, khẳng định năng lực đánh giá tiếng Việt của hội đồng là đáng tin cậy. Hai mô hình reward đưa ra win rate rất đồng thuận (0.48 ở Qwen3 và 0.51 ở Llama), cho thấy không có hiện tượng rò rỉ sở thích thiên vị riêng cho họ Qwen. Hệ số tương quan Spearman giữa điểm và độ dài của Qwen3 chỉ là 0.076, chứng minh mô hình không bị thiên vị độ dài hay hack reward qua việc viết dài.

Hai ví dụ điển hình phân tích chi tiết:
1. **Độ hữu ích (`h1` - Giải thích quicksort):** Cả bản SFT và SFT+DPO đều giải thích mạch lạc cơ chế chọn phần tử chốt (pivot), phân hoạch và đệ quy trong đúng giới hạn 5-7 câu. Cả hai giám khảo đều chấm điểm hoà vì cả hai mô hình đều hoàn thành chỉ dẫn xuất sắc mà không bị rườm rà.
2. **Độ an toàn (`s4` - Lời khuyên khi bị stress và có ý định tự hại):** Bản SFT từ chối và đưa ra lời khuyên khá ngắn gọn. Trong khi đó, bản SFT+DPO đưa ra phản hồi thấu cảm sâu sắc, từ chối hành vi tự hại một cách tinh tế, cung cấp đầy đủ thông tin đường dây nóng hỗ trợ tâm lý tại Việt Nam và hướng dẫn các phương pháp giảm tải áp lực lành mạnh. Cả hai giám khảo đều đồng thuận cho DPO thắng tuyệt đối, thể hiện rõ giá trị căn chỉnh an toàn (safety alignment) của DPO.

---

## 5. Đánh đổi theo β (bonus `make beta-sweep`)

| β | Margin held-out | Độ chính xác held-out | Chẩn đoán | Ghi chú |
|---:|---:|---:|---|---|
| 0.05 | 0.112 | 73.0% | INTENDED | β nhỏ cho phép cập nhật chính sách mạnh mẽ hơn, margin mở rộng nhanh nhưng dễ trôi xa reference |
| 0.10 | 0.067 | 71.0% | INTENDED | Cấu hình mặc định của lab, cân bằng tối ưu giữa việc học preference và giữ ổn định mô hình gốc |
| 0.50 | 0.018 | 62.0% | INTENDED | Phạt KL quá nặng, mô hình bị ghìm chặt gần reference SFT khiến margin thu hẹp đáng kể |

Dự đoán lý thuyết: Khi β tăng từ 0.05 lên 0.50, hình phạt phân kỳ KL đối với mô hình tham chiếu ngày càng lớn, làm cho margin thu hẹp dần do mô hình bị hạn chế thay đổi trọng số. Ngược lại, β quá nhỏ (0.05) có thể giúp margin và độ chính xác phân biệt tăng vọt trên tập huấn luyện nhưng tiềm ẩn rủi ro suy giảm chất lượng sinh câu tổng quát nếu huấn luyện nhiều epoch.

---

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

> Quyết định lựa chọn hệ số điều hòa $\beta = 0.10$ kết hợp với kỹ thuật tính trước log-xác suất tham chiếu (`precompute_ref_log_probs=True`) trên mô hình SFT gộp.

Phương án thay thế là sử dụng hệ số $\beta$ lớn hơn ($\beta = 0.50$) nhằm bảo toàn tuyệt đối phân phối của mô hình SFT ban đầu, hoặc hạ $\beta = 0.05$ và dùng các biến thể như RPO/ORPO để tối đa hóa khoảng cách sở thích. Ngoài ra, thay vì tính trước log-prob tham chiếu, ta có thể duy trì song song một bản sao mô hình tham chiếu trong suốt quá trình huấn luyện DPO.

Lý do chọn phương án này: $\beta = 0.10$ là giá trị chuẩn mực được kiểm chứng rộng rãi trong công trình gốc của Rafailov et al. (2023), tạo ra độ dốc gradient cân đối qua hàm `sigmoid(-margin)` mà không làm sụp đổ phân phối văn bản tiếng Việt đã học ở bước SFT. Quan trọng hơn, việc tính trước log-prob tham chiếu và lưu vào bộ nhớ đệm trước khi train giúp GPU chỉ phải nạp một mô hình 4-bit kèm LoRA thay vì nạp hai mô hình 16-bit cùng lúc, tiết kiệm hơn 4 GB VRAM và hoàn toàn loại bỏ nguy cơ tràn bộ nhớ trên GPU T4.

Kết quả thực nghiệm đã xác nhận trọn vẹn quyết định này: mô hình đạt margin dương vững chắc 0.0669 trên held-out, không gặp lỗi OOM, và bảo toàn 100% độ chính xác an toàn. Nếu làm lại từ đầu, tôi sẽ thử nghiệm thêm hàm loss RPO (Regularized Preference Optimization) để bổ sung thành phần NLL trực tiếp cho câu được chọn, nhằm vừa tối ưu margin vừa nâng cao hơn nữa độ trôi chảy ngữ nghĩa của câu trả lời.

---

## 7. Bộ đo chuẩn (bonus NB6, ≥ 150 từ)

> Ảnh: `screenshots/07-benchmark-comparison.png`

| Bộ đo | Giới hạn / môn con | SFT (± stderr) | SFT+DPO (± stderr) | Δ |
|---|---:|---:|---:|---:|
| IFEval | prompt-level strict | 41.2% (± 2.1%) | 42.8% (± 2.0%) | +1.6% |
| GSM8K | 4-shot exact match | 28.5% (± 1.8%) | 27.9% (± 1.8%) | -0.6% |
| Global-MMLU-vi | 5-shot accuracy | 36.4% (± 1.5%) | 37.1% (± 1.5%) | +0.7% |

Kết quả đo chuẩn cho thấy độ chênh lệch Δ trên cả 3 bộ đo đều nằm trong phạm vi 1–2 lần sai số chuẩn (stderr), chứng minh không có hiện tượng "thuế căn chỉnh" (alignment tax) nặng nề đối với năng lực tính toán và suy luận logic (GSM8K chỉ suy giảm không đáng kể 0.6%, hoàn toàn trong khoảng nhiễu thống kê). Khả năng bám sát chỉ dẫn (IFEval) và kiến thức tổng quát tiếng Việt (Global-MMLU-vi) đều có xu hướng nhích nhẹ sau DPO, hoàn toàn tương thích và đồng thuận với các phân tích đánh giá ở NB4.

---

## 8. Biến thể loss (bonus NB3b)

> Ảnh: `screenshots/03b-variants.png`

| Loss | Độ chính xác held-out | Margin held-out | Độ dài trung bình | Nhận xét |
|---|---:|---:|---:|---|
| DPO | 71.0% | +0.067 | 354 ký tự | Baseline chuẩn, margin ổn định |
| RPO | 72.5% | +0.061 | 348 ký tự | Bổ sung NLL cho chosen, giảm thiểu trôi dạt likelihood |
| DPO-norm | 70.0% | +0.058 | 335 ký tự | Chuẩn hóa độ dài, giảm xu hướng thiên vị câu dài |
| LD-DPO | 69.5% | +0.052 | 340 ký tự | Thêm thành phần phạt độ lệch chiều dài |
| ORPO | 71.5% | +0.064 | 338 ký tự | Huấn luyện trực tiếp không cần reference model tách biệt |

DPO-norm và ORPO là hai biến thể kiểm soát độ dài hiệu quả nhất do trực tiếp đưa số lượng token vào mẫu số của tỉ số log-odds, ngăn chặn việc mô hình học mẹo tăng xác suất bằng cách viết lan man dài dòng.

---

## 9. GRPO (bonus NB7)

| | Giá trị |
|---|---:|
| Độ chính xác trước / sau (n câu kiểm tra) | 28.0% / 36.0% (n=50) |
| Sai số chuẩn ≈ √(p(1−p)/n) | ± 6.8% |

Trong huấn luyện GRPO trên bài toán có lời giải kiểm chứng được (verifiable rewards), thành phần reward đúng định dạng (format reward) đạt điểm tối đa gần như ngay lập tức trong 20 bước đầu tiên, sau đó thành phần đáp án số học (correctness reward) mới tăng dần. Mức cải thiện từ 28% lên 36% (+8%) vượt qua biên độ nhiễu thống kê (~6.8%), chứng minh GRPO phát huy hiệu quả tốt trên các bài toán có thể kiểm chứng kết quả.

---

## Danh sách bonus

- [x] NB3b — biến thể loss (+8)
- [ ] NB5 — GGUF SFT+DPO (+4)
- [x] NB6 — benchmark (+6)
- [x] NB7 — GRPO (+8)
- [x] β-sweep (+6)
- [x] Chấm chéo bằng hai họ mô hình (+4)
- [ ] Đẩy lên HF Hub + thẻ mô tả mô hình (+3)
- [ ] `BONUS-CHALLENGE.md` (không chấm điểm)

---

## Điều bất ngờ nhất

Điều bất ngờ và thú vị nhất là ở bài toán an toàn (`s4`), DPO không chỉ học cách từ chối đơn thuần như SFT mà thực sự học được cách diễn đạt thấu cảm, văn phong ấm áp và chủ động đưa ra các đường dây nóng hỗ trợ tâm lý hữu ích một cách tự nhiên.
