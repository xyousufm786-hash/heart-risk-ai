# Data card

| | Cleveland | Pima |
|---|---|---|
| Rows (raw / clean) | 303 / 302 | 768 / 768 |
| Features | 13 (6 numeric, 7 categorical) | 8 numeric |
| Positive class | heart disease (45.7%) | diabetes (34.9%) |
| Missing after cleaning | ca 4, thal 2 | insulin 374, skin 227, BP 35, BMI 11, glucose 5 |
| Known issues | inverted label in CSV; ca=4 and thal=0 placeholders; 1 duplicate | zeros encode missing |
| Demographics available | age, sex only | women only, Pima heritage, age >= 21 |
| Intended use | teaching, benchmarking | teaching, benchmarking |
| Not suitable for | clinical decisions; claims about other populations | clinical decisions |
