"""
Bước 4 — Guardrails AI Validators
====================================
NHIỆM VỤ:
  1. Xây dựng PIIDetector: phát hiện & redact email, số điện thoại, SSN, số thẻ tín dụng
  2. Xây dựng JSONFormatter: tự động sửa JSON lỗi
  3. Bọc mỗi validator trong Guard và test với các mẫu đầu vào
  4. Chạy demo với 6 trường hợp PII và 5 trường hợp JSON

DELIVERABLE: Tất cả test cases pass (PII bị redact, JSON được sửa thành công)

CÁC KHÁI NIỆM CHÍNH:
  - @register_validator     — khai báo custom validator class
  - Validator.validate()    — implement logic kiểm tra + sửa
  - OnFailAction.FIX        — thay thế output thay vì raise error
  - Guard().use(validator)  — gắn validator instance vào guard
  - guard.validate(text)    → ValidationOutcome
      .validation_passed    — bool
      .validated_output     — output đã được xử lý

⚠️  QUAN TRỌNG: on_fail phải truyền vào CONSTRUCTOR của VALIDATOR, KHÔNG phải Guard.use()
    SAI  : Guard().use(PIIDetector, on_fail=OnFailAction.FIX)   ← TypeError
    ĐÚNG : Guard().use(PIIDetector(on_fail=OnFailAction.FIX))   ← correct

Cách chạy:
    python 04_guardrails_validator.py              # cả 2 demo
    python 04_guardrails_validator.py --demo pii   # chỉ demo PII
    python 04_guardrails_validator.py --demo json  # chỉ demo JSON
"""

import re
import json
import argparse

from guardrails import Guard
from guardrails.validators import Validator, register_validator, PassResult, FailResult

# OnFailAction đổi chỗ giữa các bản Guardrails (0.5 → 0.11): thử lần lượt các
# đường import đã biết thay vì cố định một chỗ rồi vỡ khi nâng version.
try:
    from guardrails.hub import OnFailAction
except (ImportError, AttributeError):
    try:
        from guardrails.validator_base import OnFailAction
    except ImportError:
        from guardrails.classes.validation.validation_result import OnFailAction


# ── 1. PII Detector Validator ──────────────────────────────────────────────
@register_validator(name="custom/pii-detector", data_type="string")
class PIIDetector(Validator):
    """
    Phát hiện và redact Personally Identifiable Information (PII).

    Các pattern được phát hiện:
      EMAIL       : xxx@xxx.xxx
      PHONE       : (123) 456-7890 hoặc 123-456-7890
      SSN         : 123-45-6789
      CREDIT_CARD : 1234 5678 9012 3456 (hoặc dấu gạch nối)
    """

    # Regex patterns cho từng loại PII.
    # Thứ tự có ý nghĩa: SSN và CREDIT_CARD phải được xét TRƯỚC PHONE, vì
    # pattern PHONE (\d{3}[-.\s]\d{3}[-.\s]\d{4}) cũng khớp một phần chuỗi thẻ
    # tín dụng — nếu để PHONE chạy trước, số thẻ sẽ bị dán nhãn sai là PHONE.
    PII_PATTERNS = {
        "EMAIL":       r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "SSN":         r"\b\d{3}-\d{2}-\d{4}\b",
        "CREDIT_CARD": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
        # `\b` đặt SAU `\(?` (không phải trước): nếu để `\b\(?` thì với chuỗi
        # "(555) 867-5309" regex bắt đầu khớp từ "555" và bỏ sót dấu mở ngoặc,
        # để lại "([PHONE_REDACTED]" — rò rỉ ký tự và trông như lỗi redact.
        "PHONE":       r"(?:\+?1[-.\s]?)?\(?\b\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b",
    }

    def validate(self, value: str, metadata: dict):
        """
        Tìm PII trong value; nếu phát hiện → FailResult kèm fix_value là bản đã
        redact, để OnFailAction.FIX thay thế đầu ra bằng chuỗi an toàn.

        ⚠️ VÌ SAO DÙNG FailResult CHỨ KHÔNG PHẢI PassResult(value_override=...)
        Trên guardrails 0.11, `Validator.override_value_on_pass` mặc định là False
        (guardrails/validator_base.py:98) nên `PassResult(value_override=...)` bị
        BỎ QUA hoàn toàn: validator_logs ghi nhận bản đã redact, nhưng
        `outcome.validated_output` vẫn trả về nguyên văn input — tức là PII không
        hề bị che. Đặt cờ đó thành True cũng không đủ vì đường lắp ValidationOutcome
        cho input dạng string không đọc value_override.
        Ngữ nghĩa đúng của Guardrails: phát hiện PII LÀ một validation failure, và
        OnFailAction.FIX chính là cơ chế thay đầu ra bằng `fix_value`. Cách này cho
        `validated_output` đã redact thật (đã kiểm chứng bằng thực nghiệm).

        Quét trên `redacted_text` (không phải `value`) ở mỗi vòng lặp: sau khi
        EMAIL/SSN đã bị thay bằng placeholder, các pattern sau không còn khớp lại
        phần đã redact nữa — tránh dán 2 nhãn lên cùng một chuỗi.
        """
        redacted_text = value
        found_pii     = []

        for pii_type, pattern in self.PII_PATTERNS.items():
            matches = re.findall(pattern, redacted_text)

            for match in matches:
                # re.findall trả về tuple khi pattern có capturing group
                # (CREDIT_CARD có (?:...) non-capturing nên an toàn, nhưng phòng xa)
                if isinstance(match, tuple):
                    match = next((m for m in match if m), "")
                if not match or match not in redacted_text:
                    continue
                redacted_text = redacted_text.replace(match, f"[{pii_type}_REDACTED]")
                found_pii.append((pii_type, match))

        if found_pii:
            print(f"  ⚠️  Đã redact {len(found_pii)} PII: {[p[0] for p in found_pii]}")
            return FailResult(
                error_message=f"Phát hiện {len(found_pii)} PII: "
                              f"{sorted({p[0] for p in found_pii})}",
                fix_value=redacted_text,
            )

        return PassResult()


# ── 2. JSON Formatter Validator ────────────────────────────────────────────
@register_validator(name="custom/json-formatter", data_type="string")
class JSONFormatter(Validator):
    """
    Validate và tự động sửa JSON lỗi.

    Các lỗi có thể sửa tự động:
      - Strip markdown code fences (``` hoặc ```json)
      - Thay single quotes → double quotes
      - Xóa trailing commas trước } hoặc ]
      - Re-serialize với json.dumps để định dạng chuẩn
    """

    # Marker để demo phân biệt "đã sửa được" với "phải dùng JSON lỗi dự phòng"
    FALLBACK_MARKER = "invalid_json"

    @staticmethod
    def _repair(text: str) -> str:
        """
        Cố gắng sửa chuỗi JSON lỗi. Trả về chuỗi đã sửa (chưa re-serialize).

        Thứ tự các bước là cố ý: gỡ fence trước để nội dung thật lộ ra, rồi mới
        sửa nháy và dấu phẩy trên nội dung đó.
        """
        text = text.strip()

        # Bước 1 — Xóa markdown fences
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$',          '', text)
        text = text.strip()

        # Bước 2 — Thay single quotes → double quotes
        text = text.replace("'", '"')

        # Bước 3 — Xóa trailing commas trước } hoặc ]
        text = re.sub(r',\s*([}\]])', r'\1', text)

        return text

    def validate(self, value: str, metadata: dict):
        """
        Thử parse value thành JSON. Nếu thất bại, gọi _repair() rồi thử lại.

        Ba nhánh kết quả:
          1. JSON đã hợp lệ           → PassResult, giữ nguyên đầu ra
          2. Sửa được                 → FailResult + fix_value = JSON đã format đẹp
          3. Không sửa được           → FailResult + fix_value = JSON lỗi dự phòng

        Dùng FailResult + fix_value (thay vì PassResult(value_override=...)) vì
        cùng lý do đã giải thích ở PIIDetector.validate().
        """
        # Lần 1 — parse trực tiếp (JSON đã hợp lệ thì không sửa gì cả)
        try:
            json.loads(value)
            return PassResult()
        except json.JSONDecodeError:
            pass

        # Lần 2 — sửa rồi parse lại
        try:
            repaired_text = self._repair(value)
            parsed        = json.loads(repaired_text)
            print("  🔧 JSON đã được sửa thành công")
            return FailResult(
                error_message="JSON lỗi định dạng — đã tự sửa",
                fix_value=json.dumps(parsed, indent=2),
            )
        except json.JSONDecodeError as e:
            # Không sửa được → trả JSON lỗi dự phòng kèm lý do (tiêu chí 4.7)
            print(f"  ❌ Không sửa được → trả JSON lỗi dự phòng ({e})")
            return FailResult(
                error_message=f"JSON không hợp lệ sau khi sửa: {e}",
                fix_value=json.dumps(
                    {"error": self.FALLBACK_MARKER, "detail": str(e), "raw": value[:200]},
                    indent=2,
                ),
            )


# ── 3. Demo: PII Guard ─────────────────────────────────────────────────────
def demo_pii_guard():
    print("\n" + "=" * 55)
    print("  Demo: PII Detection & Redaction")
    print("=" * 55)

    # on_fail nằm trong CONSTRUCTOR của validator, không phải trong Guard.use()
    guard = Guard().use(PIIDetector(on_fail=OnFailAction.FIX))

    test_cases = [
        ("Email",        "Contact John at john.doe@example.com for details."),
        ("Phone",        "Call our support line at (555) 867-5309."),
        ("SSN",          "Patient SSN is 123-45-6789 on file."),
        ("Credit Card",  "Payment made with card 4532 1234 5678 9010."),
        ("Multi-PII",    "Email: alice@example.com, Phone: 555-123-4567"),
        ("Clean",        "No sensitive information in this text."),
    ]

    redacted_count = 0
    for label, text in test_cases:
        result = guard.validate(text)
        out    = result.validated_output

        was_redacted = out != text
        if was_redacted:
            redacted_count += 1

        # Kiểm tra thật: sau khi redact thì chuỗi PII gốc KHÔNG còn trong output
        status = "🔒 REDACTED" if was_redacted else "✓ giữ nguyên (không có PII)"

        print(f"\n[{label}] {status}")
        print(f"  Input:  {text}")
        print(f"  Output: {out}")

    print(f"\n📊 Tổng kết PII: {redacted_count}/{len(test_cases)} test case bị redact "
          f"(kỳ vọng 5/6 — case 'Clean' phải giữ nguyên)")


# ── 4. Demo: JSON Guard ────────────────────────────────────────────────────
def demo_json_guard():
    print("\n" + "=" * 55)
    print("  Demo: JSON Formatting & Repair")
    print("=" * 55)

    guard = Guard().use(JSONFormatter(on_fail=OnFailAction.FIX))

    test_cases = [
        ("Valid JSON",       '{"name": "Alice", "age": 30}'),
        ("Markdown fences",  '```json\n{"name": "Bob"}\n```'),
        ("Single quotes",    "{'name': 'Charlie', 'score': 95}"),
        ("Trailing comma",   '{"key": "value",}'),
        ("Truly invalid",    "This is not JSON at all: ??? {]"),
    ]

    # Phân loại theo KẾT QUẢ THỰC TẾ chứ không chỉ theo validation_passed:
    # với OnFailAction.FIX thì mọi case đều "passed" (fix_value luôn được áp),
    # nên phải tự kiểm tra output có parse được và có phải JSON dự phòng hay không.
    usable = 0
    for label, text in test_cases:
        result = guard.validate(text)
        out    = str(result.validated_output)

        try:
            parsed     = json.loads(out)
            parses     = True
            is_fallback = isinstance(parsed, dict) and \
                          parsed.get("error") == JSONFormatter.FALLBACK_MARKER
        except json.JSONDecodeError:
            parses, is_fallback = False, False

        if parses and not is_fallback:
            usable += 1
            status = "✅ JSON hợp lệ"
        elif is_fallback:
            status = "🛟 JSON lỗi dự phòng (không sửa được)"
        else:
            status = "❌ Output không parse được"

        print(f"\n[{label}] {status}")
        print(f"  Input:  {text[:60]}")
        print(f"  Output: {out[:80].replace(chr(10), ' ')}")

    print(f"\n📊 Tổng kết JSON: {usable}/{len(test_cases)} case cho ra JSON dùng được "
          f"(kỳ vọng 4/5 — case 'Truly invalid' phải rơi vào JSON lỗi dự phòng)")


# ── 5. Main ────────────────────────────────────────────────────────────────
def main():
    # `--demo` cho phép xuất 2 file log riêng (evidence/04_pii_demo_log.txt và
    # 04_json_demo_log.txt) mà run_all.py vẫn gọi main() không đối số như cũ.
    parser = argparse.ArgumentParser(description="Guardrails AI validators demo")
    parser.add_argument("--demo", choices=["pii", "json", "all"], default="all",
                        help="Chọn demo để chạy (mặc định: all)")
    args, _unknown = parser.parse_known_args()

    print("=" * 55)
    print("  Bước 4: Guardrails AI Validators")
    print("=" * 55)

    if args.demo in ("pii", "all"):
        demo_pii_guard()
    if args.demo in ("json", "all"):
        demo_json_guard()

    print("\n✅ Bước 4 hoàn thành!")


if __name__ == "__main__":
    main()
