;prick=en tidsenhet
;streck=tre tidsenheter
;mellan signlar= en tidsenhet
;mellan bokstäver=tre tidsenheter
;mellan ord= 7 tidsenheter
;en tidsenhet=50 ms

;subrutiner:
;Översätt ett tecken till Morsekod 		LOOKUP	$2100
;Vänta en halv period 				DELAY	$2400
;Sänd ett streck eller en prick 			BEEP	$2300
;Sänd ett tecken 				SEND	$2200
;Huvudprogram: starta, sänd ut sträng, avsluta 	MAIN	$2000

;variabler:
;Sträng $6000
;Morsetabell $7000
;T för DELAY $7100
;N för BEEP $7104
;Stackpekare $8000

MAIN: 	LEA $8000,A7		; Sätt stackpekaren
	CLR.B	$10084		; Programmera PIA: nollställ CRA
	MOVE.B	#$00,$10080	; och nollställ CRB
	MOVE.B	#$04,$10084	; Programmera DDRA och DDRB.
	CLR.B	$10086		; En bit utgång, resten ingångar
	MOVE.B	#$FF,$10082
	MOVE.B	#$04,$10086

	MOVE.B #$04,$10084 	; Inga avbrott skall genereras,
	MOVE.B #$04,$10086 	; varken från port A eller port B

	BCLR	#7,$10082		; Sätt utbiten till 0
	LEA $6000,A0 		; Sätt A0 till att peka på strängen
	MOVE.L #109,$7100	;sätter frekvensen till 500hz
	MOVE.L #50,$7104	;sätter antalet perioder till 25
	

NEXTCH: MOVE.B (A0)+,D0 	; Hämta nästa tecken
	CMP.B #0,D0		; Om D0=0: sträng slut, gå till END
	BEQ END			; Annars: anropa LOOKUP
	JSR LOOKUP		
	CMP.B #0,D0		; Kolla det returnerade värdet i D0
	BEQ BLANK 		; Om D0=0: gå till NEXTCH
	JSR SEND		; Annars: anropa SEND
	JSR NEXTCH			; Gå till NEXTCH för nästa tecken
END: 	MOVE.B #228,D7 		; Gå tillbaka till
	TRAP #14 		; monitorn i TUTOR

;ascii-koden ska ligga i D0 innan anrop
LOOKUP: LEA $7000,A1 		; Peka på tabellen
	AND.W #$00FF,D0 	; Gör D0 till 16 bitar
	MOVE.B (A1,D0.W),D0 	; Hämta Morsekoden
	RTS

;lägg tidsförfröjningen på adressen $7100 innan,218 ger ca 1 ms
DELAY: 	MOVE.L $7100,D3 	; Hämta fördröjning T
WAIT: 	SUBQ.L #1,D3 		; Räkna ned med 1
	BNE WAIT 		; Om D3<>0: räkna vidare
	RTS 			; Hoppa tillbaka

BEEP: 	CMP.B #1,D2		; Om D2=1: ettställ utbiten,
	SEQ $10082		; annars nollställ utbiten.
				; (Tips: instruktionen Scc!)
	BSR DELAY 		; Vänta en halv period
	BCLR #7,$10082		; Nollställ utbiten
	BSR DELAY 		; Vänta en halv period
	SUBQ.L #1,D1 		; Räkna ned D1
	BNE BEEP 		; Om D1>0: fortsätt pipa
	RTS

SEND: 	MOVE.L $7104,D1 	; Hämta antal perioder N för prick
	LSL.B #1,D0 		; Skifta upp nästa symbol i Morsekoden
	BEQ READY 		; Om D0=0 är tecknet slut
	BCC DOT 		; C=0: prick, C=1: streck
	DASH: MULU #3,D1 	; Multiplicera D1 med 3
DOT: 	MOVE.B #1,D2 		; Ladda D2 med 1 för ton
	BSR BEEP 		; Sänd ut prick/streck
	MOVE.L $7104,D1 	; Hämta N igen
	MOVE.B #0,D2 		; Ladda D2 med 0 för paus
	BSR BEEP 		; Sänd paus efter prick/streck
	BRA SEND 		; Sänd ut nästa prick/streck
	READY: ASL.L #1,D1 	; Öka D1 till 2N
	MOVE.B #0,D2 		; Ladda D2 med 0 för paus
	BSR BEEP 		; Sänd ut extra paus efter tecknet
	RTS 			; Hoppa tillbaka från subrutinen
BLANK
	MOVE.L $7104,D1
	ASL #2,D1
	MOVE.B #0,D2
	BSR BEEP
	BRA NEXTCH
