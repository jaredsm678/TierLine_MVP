# Smoke raw outputs — 4 OpenRouter models x 5 difficult rows (V1)

Captured by `scripts/smoke.py` (raw = the model's returned message content;
thinking is stripped by the provider per the frozen design §7.2/§8).

## Claude 5.5 Opus Thinking, Max Effort (`claude-5.5-opus-thinking-V1.jsonl`)

| row | gold intent | prediction | correct | oos sel | latency | cost | raw |
|---|---|---|---|---|---|---|---|
| bnk-0001 | account_blocked | account_blocked | Y | N | 7330ms | $0.0151 | `account_blocked` |
| bnk-0011 | bill_balance | bill_balance | Y | N | 7120ms | $0.0139 | `bill_balance` |
| bnk-0021 | freeze_account | freeze_account | Y | N | 3636ms | $0.0045 | `freeze_account` |
| bnk-0031 | min_payment | min_payment | Y | N | 3328ms | $0.0047 | `min_payment` |
| bnk-0041 | pay_bill | pay_bill | Y | N | 3374ms | $0.0045 | `pay_bill` |

**5/5 correct, 0 false abstention(s), $0.0427**

## Gemini 3.8 Flash, High (`gemini-3.8-flash-V1.jsonl`)

| row | gold intent | prediction | correct | oos sel | latency | cost | raw |
|---|---|---|---|---|---|---|---|
| bnk-0001 | account_blocked | out_of_scope | N | Y | 13983ms | $0.0078 | `out_of_scope` |
| bnk-0011 | bill_balance | bill_balance | Y | N | 9012ms | $0.0044 | `bill_balance` |
| bnk-0021 | freeze_account | freeze_account | Y | N | 3375ms | $0.0010 | `freeze_account` |
| bnk-0031 | min_payment | min_payment | Y | N | 3692ms | $0.0018 | `min_payment` |
| bnk-0041 | pay_bill | pay_bill | Y | N | 2586ms | $0.0012 | `pay_bill` |

**4/5 correct, 1 false abstention(s), $0.0162**

## GPT-6.1 Sol, Max Effort (`gpt-6.1-sol-V1.jsonl`)

| row | gold intent | prediction | correct | oos sel | latency | cost | raw |
|---|---|---|---|---|---|---|---|
| bnk-0001 | account_blocked | out_of_scope | N | Y | 6593ms | $0.0038 | `out_of_scope` |
| bnk-0011 | bill_balance | bill_balance | Y | N | 7115ms | $0.0040 | `bill_balance` |
| bnk-0021 | freeze_account | freeze_account | Y | N | 3246ms | $0.0017 | `freeze_account` |
| bnk-0031 | min_payment | min_payment | Y | N | 4225ms | $0.0025 | `min_payment` |
| bnk-0041 | pay_bill | pay_bill | Y | N | 3378ms | $0.0019 | `pay_bill` |

**4/5 correct, 1 false abstention(s), $0.0139**

## Kimi K3 (`kimi-k3-V1.jsonl`)

| row | gold intent | prediction | correct | oos sel | latency | cost | raw |
|---|---|---|---|---|---|---|---|
| bnk-0001 | account_blocked | out_of_scope | N | Y | 3393ms | $0.0033 | `out_of_scope` |
| bnk-0011 | bill_balance | out_of_scope | N | Y | 2454ms | $0.0015 | `out_of_scope` |
| bnk-0021 | freeze_account | freeze_account | Y | N | 2355ms | $0.0017 | `freeze_account` |
| bnk-0031 | min_payment | min_payment | Y | N | 1942ms | $0.0016 | `min_payment` |
| bnk-0041 | pay_bill | pay_bill | Y | N | 2212ms | $0.0021 | `pay_bill` |

**3/5 correct, 2 false abstention(s), $0.0103**
