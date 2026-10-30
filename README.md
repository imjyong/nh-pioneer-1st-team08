<div align="center">

# 🥷 꾸러기닌자토리

**NH선물 REST API Pioneer 1기 · Team 08**

주차별 과제와 프로젝트 진행 과정을 정리하는 저장소입니다.

<br/>

<table>
  <tr>
    <td align="center" width="260">
      <a href="https://github.com/chloemjshin1"><img src="https://github.com/chloemjshin1.png" width="100" alt="신민정"/></a><br/>
      <b>신민정</b> <a href="https://github.com/chloemjshin1">@chloemjshin1</a><br/>
      <sub>전략 설계, 시장 국면 분류,<br/>검증 및 리스크 관리 기준</sub>
    </td>
    <td align="center" width="260">
      <a href="https://github.com/imjyong"><img src="https://github.com/imjyong.png" width="100" alt="황지영"/></a><br/>
      <b>황지영</b> <a href="https://github.com/imjyong">@imjyong</a><br/>
      <sub>데이터 처리, WebSocket 실시간 시세 연동,<br/>주문 시스템 개발</sub>
    </td>
  </tr>
</table>

<img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/REST%20API-005571?style=flat-square&logo=fastapi&logoColor=white"/>
<img src="https://img.shields.io/badge/WebSocket-010101?style=flat-square&logo=socketdotio&logoColor=white"/>
<img src="https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white"/>

</div>

## 관심 분야

- KOSPI200 선물 추세추종
- HMM 기반 시장 국면 분류와 국면별 포지션 조절
- 주문 및 리스크 관리 자동화
- 통화선물

## 저장소 구조

```text
core/                 # API 요청, 시세 조회, 지표, 주문 등 공통 코드
weekNN/
  README.md           # 해당 주차 보고서
  *.py, config.json   # 전략, 백테스트, 장중 실행 코드와 설정
  results/            # 실행 로그
  screenshots/        # 체결 내역과 잔고 화면
requirements.txt      # 필요한 패키지
.env.example          # API 키 입력 양식
```

## 주차별 일정

| 주차 | 제출일 | 주제 | 상품 & 거래소 | 전략 | 상태 |
| :---: | :---: | --- | --- | --- | :---: |
| [01](week01/README.md) | 10/23 | API 키 발급, 팀 GitHub 개설 | 코스피200 선물 (KRX) | – | 🟢 |
| [02](week02/README.md) | 10/30 | 국내상품 전략 구현 | 3년 국채선물 2026년 12월물 (KRX) | 5분봉 SMA 12/36 교차, ATR × 2.5 손절 | 🟢 |
| 03 | 11/06 | 해외상품 및 거래소 전략 구현 | | | ⚪ |
| 04 | 11/13 | 복수상품 및 스프레드 전략 | | | ⚪ |
| 05 | 11/20 | 데이터 기반 전략 | | | ⚪ |
| 06 | 11/27 | 주문 및 리스크 관리 응용 | | | ⚪ |
| 07 | 12/04 | 최종 프로젝트 1 | | | ⚪ |
| 08 | 12/11 | 최종 프로젝트 2 | | | ⚪ |

<sub>🟢 제출 완료 &nbsp; 🟡 진행 중 &nbsp; ⚪ 예정 &nbsp;&nbsp;
