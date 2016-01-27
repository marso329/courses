;$900 innehåller sekund ental
;$901 innehåller sekund tiotal
;$902 innehåller minut ental
;$902 innehåller minut tiotal
START
	CLR.B	$900		;nollställer siffervärden
	CLR.B	$901		;nollställer siffervärden
	CLR.B	$902		;nollställer siffervärden
	CLR.B	$903		;nollställer siffervärden
	
	MOVE.B #$3F,$7000	;0
	MOVE.B #$06,$7001	;1
	MOVE.B #$5B,$7002	;2
	MOVE.B #$4F,$7003	;3
	MOVE.B #$66,$7004	;4
	MOVE.B #$6D,$7005	;5
	MOVE.B #$7D,$7006	;6
	MOVE.B #$07,$7007	;7
	MOVE.B #$7F,$7008	;8
	MOVE.B #$67,$7009	;9

PIAINIT
				;sätter subrutinen mux att triggas då en level 2 interrupt sker
	MOVE.L	#BCD,$74
				; sätter subrutinen bcd att triggas då en level 5 interrupt sker
	MOVE.L	#MUX,$68
	CLR.B $10084 		; Nollställ CRA
	MOVE.B #$7F,$10080 	; A7=in,A6-A0=ut
	MOVE.B #$07,$10084 	;CRA-7 1-ställs då CA1 0->1,IRQA triggars då, piaa utpekas(nivå 5)

	CLR.B $10086 		; Nollställ CRB
	MOVE.B #$03,$10082 	; B7-B2=in,B1-B0=ut
	MOVE.B #$07,$10086	;CRB-7 1-ställs då CB1 0->1,IRQB triggars då, piab utpekas(nivå 2), det var en 4 tidigare
	MOVE.L #$8000,A7
	MOVE.L #0000,$7104
	AND.W 	#$F8FF,SR
	
LOOP	
	JMP	LOOP
		
MUX
	TST.B	$10082
	MOVE.L #0,D1
	;MOVE.B $7104,D1
	LEA $7104,A5
	MOVE.L (A5),D1
	LEA $900,A0	;lägger adressen till första värdet i A0
	ADD.W	D1,A0		; Plussar 
	MOVE.B	D1,$10082	;sätter muxen
	ADD.W	#$0001,D1	;PLUSSER ETT
	CMP.W 	#$0004,D1
	BEQ	RESET

MUX2	MOVE.L D1,$7104		; SKRIVER TILL MINNET FÖR RÄKNAREN
	CLR.L	D0
	MOVE.B	(A0),D0		;lägger det nuvarande siffervärdet i d0
	LEA 	$7000,A1	;A1 pekar på tabellen
	ADD.L 	D0,A1		;lägger på det nuvarande siffervärdet på adressen till kodningen
	MOVE.B 	(A1),D0 	;hämtar segment kodning
	MOVE.B	D0,$10080	;skriver siffervärdet till utgången
	RTE			;är det inte uppfyllt så hoppar den tillbaka från avbrottet
RESET
	MOVE.L 	#0,D1	
	JMP 	MUX2
	


BCD
	TST.B	$10080
	MOVE.B 	$900,D0		;flytter sekunder ental till d0
	MOVE.B 	$901,D1		;flyttar sekunder tiotal till d1
	MULU.W 	#10,D1		;multiplicerar tiotal med 10
	ADD.B	D1,D0		;adderar tiotal och ental och lägger det i d0
	ADD.B	#1,D0		;plussar på ett
	CMP.B	#60,D0		;om antalet sekunder är mindre än 60 så hoppar den till JUMP
	BNE	JUMP		;
	MOVE.B 	#0,$900		;annars så nollar den sekunderna
	MOVE.B 	#0,$901		;
	MOVE.B 	$902,D0		;flyttar minuter ental till d0
	MOVE.B 	$903,D1		;flyttar sekunder tiotal till d1
	MULU.W 	#10,D1		;multiplicerar värdet i d1 med 10
	ADD.B	D1,D0		;d1+d0->d0
	ADD.B	#1,D0		;d0=d0+1
	CMP.B	#60,D0		;om antalet minuter är 60 så hoppar den till jump1
	BEQ	JUMP1	
	MOVE.W	#$0000,D3
	MOVE.B 	D0,D3		;kopierar d0 till d3
	DIVU.W	#10,D0		;dividerar d0 med 10
	MOVE.B	D0,$903		;flyttar värdet i d0 till sekunder tiotal
	MULU.W 	#10,D0		;multiplicerar värdet i d0 med 10
	SUB.B	D0,D3		;d3-d0 vilket ger sekunder ental
	CMP.B	#$0A,D3
	BEQ	JUMP4
	MOVE.B	D3,$902		;flyttar värdet till minnet för sekunder ental
	RTE			;hoppar tillbaka från avbrottet
	
JUMP1	
	MOVE.B 	#0,$902	;nollar minuter ental
	MOVE.B 	#0,$903		;nollar minuter tiotal
	RTE			;hoppar tillbaka från avbrottet
JUMP
	MOVE.W	#$0000,D3
	MOVE.B 	D0,D3		;kopierar d0 till d3
	DIVU.W	#10,D0		;dividerar d0 med 10
	MOVE.B	D0,$901		;flyttar värdet i d0 till sekunder tiotal
	MULU.W 	#10,D0		;multiplicerar värdet i d0 med 10
	SUB.B	D0,D3		;d3-d0 vilket ger sekunder ental
	CMP.B	#$0A,D3
	BEQ	JUMP3
	MOVE.B	D3,$900		;flyttar värdet till minnet för sekunder ental
	RTE			;hoppar tillbaka från avbrottet
JUMP3
	MOVE.B 	#0,$900
	RTE
JUMP4
	MOVE.B 	#0,$902
	RTE

