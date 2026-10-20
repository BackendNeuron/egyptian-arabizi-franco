# Arabic spelling style (the targets)

The model learns whatever the Arabic side does, so every target follows one spelling. Loosely based on CODA
(the Conventional Orthography for Dialectal Arabic used by the Egyptian BOLT corpus), simplified for chat.

| Case | Always write | Never |
|---|---|---|
| "what" | ايه | إيه، اية |
| feminine ending | ة (بكرة، حلوة، الساعة) | ه for ة |
| final ى vs ي | ي for the sound "i/ee" (في، علي "on me"), ى only in standard words (على، مستشفى، حتى) | |
| hamza at word start | ا (انا، احنا، امبارح) | أ، إ — chat almost never types them |
| "the" | ال attached, sun letters written as is (الشمس) | |
| future | ه attached (هروح، هنشوف) | ح |
| negation | ما... ش attached (مش، مبحبش، ماعرفش) | |
| "this" | ده، دي، دول | |
| Everyday loanwords Egyptians write in Arabic | Arabic: موبايل، تاكسي، اوبر، بيتزا، فيلم، فيديو، بوست، لايك، ستوري، ماتش | Latin |
| Work and tech English words | kept in Latin letters, Arabic ال attached: الmeeting، الlink، الfile، الdeadline، الcode، الpassword | transliterated |
| numbers | digits stay digits: 5، 2026 | |
| emojis and punctuation | unchanged | |

Diacritics (تشكيل) are never written. Tatweel (ـ) and stretched letters are removed from targets.
