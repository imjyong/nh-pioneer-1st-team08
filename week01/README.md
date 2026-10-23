# Week 01. API 키 발급 및 팀 GitHub 개설

## 수행 내용

써핑보드 HTS에서 국내선물 모의투자용 APP Key와 Secret Key를 발급받고, REST API 개발자센터에서 접근토큰을 발급받아 현재가 조회 기능이 정상적으로 동작하는지 확인했습니다.

## To Do

- [x] APP Key, Secret Key 발급
- [x] API 호출 테스트
- [x] 팀 GitHub 저장소 생성
- [x] README 작성
- [x] Discord #team-08 채널에 저장소 링크 제출

## API 호출 테스트

| 항목 | 내용 |
| --- | --- |
| 테스트 환경 | 국내선물 모의투자 |
| 테스트 종목 | 코스피200 F 202612 (`A016C000`) |
| 테스트 방법 | NH선물 REST API 개발자센터 API 테스트 |
| 인증 결과 | 접근토큰 정상 발급 (`HTTP 200`) |
| 조회 결과 | 발급받은 토큰으로 현재가 정상 조회 (`HTTP 200`) |

```
POST /auth-service/v1/token   # 접근토큰 발급
POST /trade/v1/domestic/price # 국내 현재가 조회
```
