from pathlib import Path

from src.metric import aliment


sample_folder = Path("exp_local/phish-email/2025.12.19/105001/samples")

# 初始化metric
metric = aliment.make_default_alignment_metric(
    model_name="intfloat/e5-base-v2",
    use_sentence_transformers=True,
    policy=aliment.MaxSimPolicy(),  # 或 TopKMeanPolicy(k=3)
)

# # 获取所有iter_xxx目录，按照数字大小排序
# iter_dirs = sorted(
#     [d for d in os.listdir(sample_folder) if d.startswith("iter_")],
#     key=lambda x: int(x.replace("iter_", ""))
# )

# results = []

# for iter_dir in iter_dirs:
#     iter_path = sample_folder / iter_dir

#     # 获取sample_0.json和sample_1.json
#     sample_files = [
#         iter_path / "sample_0.json",
#         iter_path / "sample_1.json",
#     ]

#     captions = []
#     emails = []

#     # 读取所有样本
#     for file in sample_files:
#         if file.exists():
#             with open(file, "r") as f:
#                 samples = json.load(f)
#             for entry in samples:
#                 captions.append(entry["caption"])
#                 emails.append(entry["text"])

#     # 计算该iter下的scores
#     scores = []
#     for caption, email in zip(captions, emails):
#         score = metric.score(caption, email)
#         scores.append(score)

#     # 计算平均score
#     if scores:
#         avg_score = sum(scores) / len(scores)
#         results.append({
#             "iter": iter_dir,
#             "num_samples": len(scores),
#             "avg_score": avg_score,
#         })
#         print(f"{iter_dir}: {len(scores)} samples, avg_score: {avg_score:.4f}")

# # 打印总结
# print("\n=== Summary ===")
# for result in results:
#     print(f"{result['iter']}: {result['num_samples']} samples, avg_score: {result['avg_score']:.4f}")

caption_email_pairs = [
    {
        "alignment": "very_low",
        "caption": "Live your best life. Every day is a new beginning.",
        "email": (
            "Subject: Q2 Product Launch Schedule\n\n"
            "Dear Partners,\n\n"
            "We are writing to confirm that our Q2 product launch event "
            "will take place on June 15. The session will focus on pricing, "
            "distribution policies, and technical specifications.\n\n"
            "Please confirm your attendance.\n\n"
            "Best regards,\n"
            "Marketing Team"
        ),
        "note": "Caption is emotional and generic; email is formal and transactional with no shared message.",
    },
    {
        "alignment": "low",
        "caption": "A new chapter begins. Stay tuned for what’s next.",
        "email": (
            "Subject: Update on Upcoming Product Release\n\n"
            "Hello,\n\n"
            "Our upcoming product includes several improvements in system "
            "performance and reliability. Detailed documentation will be "
            "shared separately.\n\n"
            "Regards,\n"
            "Product Team"
        ),
        "note": "Both imply change, but caption is vague and does not clearly lead into the email content.",
    },
    {
        "alignment": "medium",
        "caption": "Something new is coming—designed to work smarter.",
        "email": (
            "Subject: Introducing New Efficiency Features\n\n"
            "Dear Customer,\n\n"
            "We are pleased to introduce new features focused on improving "
            "operational efficiency and ease of use. These updates are part "
            "of our upcoming release.\n\n"
            "More details will follow.\n\n"
            "Sincerely,\n"
            "Customer Success Team"
        ),
        "note": "Caption and email share a general theme, but the connection is still indirect.",
    },
    {
        "alignment": "high",
        "caption": "Smarter workflows start here. Meet our latest upgrade.",
        "email": (
            "Subject: Our Latest Upgrade Is Here\n\n"
            "Hello,\n\n"
            "Today we’re excited to announce our latest upgrade, built to "
            "streamline workflows and reduce manual effort. Key highlights "
            "include faster setup and improved system visibility.\n\n"
            "Thank you for continuing the journey with us.\n\n"
            "Best,\n"
            "Product Marketing"
        ),
        "note": "Caption sets a clear expectation that the email fully delivers on.",
    },
    {
        "alignment": "very_high",
        "caption": "Introducing our latest upgrade—built for faster, smarter workflows.",
        "email": (
            "Subject: Introducing Our Latest Upgrade\n\n"
            "Dear Valued Customer,\n\n"
            "We are proud to introduce our latest upgrade, designed specifically "
            "to enable faster, smarter workflows. This release focuses on "
            "automation, clarity, and performance improvements requested by our users.\n\n"
            "Explore the full feature set and see how it supports your daily operations.\n\n"
            "Kind regards,\n"
            "The Product Team"
        ),
        "note": "Caption and email are tightly aligned in language, intent, and value proposition.",
    },
]

for pair in caption_email_pairs:
    score = metric.score(pair["caption"], pair["email"])
    print(f"Alignment: {pair['alignment']}, Score: {score:.4f} -- {pair['note']}")
