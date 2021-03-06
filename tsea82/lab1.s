PIAINIT
	CLR.B	$10084
	MOVE.B	#$00,$10080
	MOVE.B	#$04,$10084

	CLR.B	$10086
	MOVE.B	#$FF,$10082
	MOVE.B	#$04,$10086

START	MOVE.B	#0,D1
	CLR	D2
	CLR	D3
	MOVE.B	#1,D2
IDLE
	MOVE.B 	$10080,D0
	AND.B 	#$01,D0
	BEQ	IDLE
DELAY	
	MOVE.L	#32,D1		;change to T/2,changed from 57 to 1
DEL2	BSET	#7,$10082
	SUB.L	#1,D1
	BNE.S	DEL2
	BCLR	#7,$10082

	MOVE.B 	$10080,D0
	AND.B 	#$01,D0
	BEQ	IDLE
		
	CLR	D0
	LEA	FETCH,A0
	JMP	DELAY2
FETCH
	BSET	#7,$10082
	MOVE.B 	$10080,D0
	BCLR	#7,$10082
	AND.B 	#$01,D0
	BEQ	TEMP
	ADD.L	D2,D3
TEMP	MULS	#2,D2
	CMP.L	#16,D2
	BEQ	SCREEN
	JMP	DELAY2

SCREEN
	MOVE.B	D3,$10082
	JMP	DELAY4
	
DELAY2	
	MOVE.L	#66,D1		;change to T,changed from 57 to 1
DEL4	BSET	#7,$10082
	SUB.L	#1,D1
	BNE.S	DEL4
	BCLR	#7,$10082
	JMP	(A0)

DELAY4	
	MOVE.L	#64,D1		;change to A BIG NUMBER,changed from 57 to 1
DEL8	BSET	#7,$10082
	SUB.L	#1,D1
	BNE.S	DEL8
	BCLR	#7,$10082
	JMP	START
