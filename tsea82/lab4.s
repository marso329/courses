
; PIA:n skall anslutas/konfigureras enligt följande:
; PIAA b7-b6 A/D-omvandlare i X-led
;b5-b4A/D-omvandlare i Y-led
;b3Används inte
;b2-b0Styr diodmatrisens multiplexingångar
;
; CA b2 Signal till högtalare
;b1Avbrottssignal för MUX-rutinen
;
; PIAB b7 Används inte
;b6-b0Diodmatrisens bitmönster
;
; CB b2 Starta omvandling hos A/D-omvandlare
;b1Används inte
; jump to program

	jmp COLDSTART
POSX	dc.b 0 			; rightmost column (0..6)
POSY 	dc.b 0 			; middle row (0..4)
				; fixed target position
TPOSX 	dc.b 0 			; target position (0..6)
TPOSY 	dc.b 0 			; target position (0..4)
				; line shown for multiplexing
LINE 	dc.b 0 			; current line 0..4 shown on display
				; random number
RND 	dc.b 0 			; random num
COLDSTART
	MOVE.L #$8000,A7	; set stack pointer
	jsr PIAINIT 		; setup I/O
	jsr INSTALLINTS 	; install and enable interrupts
				; short CB1 to GND for now unless
				; you really want interrupts
WARMSTART
	move.b #0,POSX 		; we always start from here
	move.b #2,POSY 		;
	jsr RANDOMTARGET 	; positon target
GAME
				; sense joystick and update POSX, POSY
	jsr JOYSTICK
				; update videomem with POSX, POSY and target
	jsr VIDEOINIT 		; clear it to draw a new frame
	move.b POSY,d0
	and.l #$000000ff,d0
	lea $900,a0
	add.l d0,a0
	move.b POSX,d0
	bset d0,(a0)
				; target position also
	move.b TPOSY,d0
	and.l #$7,d0
	lea $900,a0
	add.l d0,a0
	move.b TPOSX,d0
	bset d0,(a0)
				; wait a bit
	move.l #10000,d7
DLY 	sub.l #1,d7
	bne DLY
				; analyze situation
	
	MOVE.B	POSX,D2
	MOVE.B	TPOSX,D1			; we have a hit if POSX=TPOSX and POSY=TPOSY
	CMP.B D2,D1			;skriv rutinen som kollar om vi har träff ***
	BNE GAME
	MOVE.B	POSY,D2
	MOVE.B	TPOSY,D1			; we have a hit if POSX=TPOSX and POSY=TPOSY
	CMP.B D2,D1
	BNE GAME			;om inte träff börjar programmet om från GAME ***
				; we have a hit! Sound the alarm!
	jsr BEEP
				; and restart
	jmp WARMSTART
				;
				; Joystick sensing routine
				; also sets X- and Y-coords
				;

JOYSTICK
				;starta en omvandling hos A/D-omvandlarna **
	BCLR	#3,$10086	
	MOVE.L	#1,D1		;VÄNTAR ETT TAG
DEL1	
	SUB.L	#1,D1
	BNE.S	DEL1
	BSET	#3,$10086
	MOVE.L	#1,D1		;VÄNTAR ETT TAG
DEL2	
	SUB.L	#1,D1
	BNE.S	DEL2
XCOORD
	move.b $10080,d0 	; read both A/D:s
	AND.W #$C0,D0
	LSR.B #6,D0
	CMP.B	#3,D0		;om d0 är 3
	BEQ	INCREASEX
	CMP.B	#0,D0		;om d0 är 3
	BEQ	DECREASEX	
YCOORD
	move.b $10080,d0 	; what was it now again?
	AND.W #$30,D0
	LSR.B #4,D0
	CMP.B	#3,D0		;om d0 är 3
	BEQ	INCREASEY
	CMP.B	#0,D0		;om d0 är 3
	BEQ	DECREASEY
BACK	
	jsr LIMITS 		; bounds check before leaving
	rts
				; LIMITS keeps us from falling off the edge of the world
				; Allowed: POSX 0..6
				; POSY 0..4
DECREASEX
	add.b #1,POSX
	JMP	YCOORD
INCREASEX
	sub.b #1,POSX
	JMP	YCOORD
INCREASEY
	add.b #1,POSY
	JMP BACK
DECREASEY
	sub.b #1,POSY
	JMP BACK
	
LIMITS
	move.b POSX,d0 		; get current (updated) X-coord
	bpl LIM1 		; too much to right?
	move.b #0,POSX 		; not any longer
LIM1
	cmp.b #7,d0 		; too much to left?
	bne LIMY 		; nope, check Y-coord
	move.b #6,POSX 		; stick to left border
LIMY
	move.b POSY,d0 		; get current (updated) Y-coord
	bpl LIM2 		; below arena?
	move.b #0,POSY 		; keep on arena
LIM2
	cmp.b #5,d0 		; above arena?
	bne LIM_EXIT 		; no.
	move.b #4,POSY 		; keep on arena
LIM_EXIT
				; both coords within bounds here
	rts 			; done
				;
				; Interrupt routine for multiplexing
				; Installed as IRQA
				;
MUX
	MOVE.L	D1,-(A7)
	MOVE.L	A5,-(A7)
	MOVE.L	D0,-(A7)
	MOVE.L	A0,-(A7)
	TST.B	$10080		;behövs tydligen
	MOVE.L #0,D1		;skriver 0 till d1
	LEA $7108,A5		;låter a5 peka på räknaren
	MOVE.L (A5),D1		;hämtar räknaren till d1
	LEA $900,A0		;lägger adressen till första värdet i A0
	ADD.W	D1,A0		; Plussar 
	MOVE.B	#0,$10082
	MOVE.B	D1,$10080	;sätter muxen
	ADD.W	#$0001,D1	;PLUSSER ETT
	CMP.W 	#$0005,D1
	BEQ	RESET

MUX2	
	MOVE.L D1,$7108		; SKRIVER TILL MINNET FÖR RÄKNAREN
	CLR.L	D0
	MOVE.B	(A0),D0		;lägger det nuvarande siffervärdet i d0
	MOVE.B	D0,$10082	;skriver siffervärdet till utgången
	add.b #1,RND
	MOVE.L	(A7)+,A0
	MOVE.L	(A7)+,D0
	MOVE.L	(A7)+,A5
	MOVE.L (A7)+,D1
	RTE			;är det inte uppfyllt så hoppar den tillbaka från avbrottet
RESET
	MOVE.L 	#0,D1	
	JMP 	MUX2
				;
				; Videoinit clears video mem
				;
VIDEOINIT
	clr.b $900 		; clear memory
	clr.b $901 		; ... ditto
	clr.b $902 		;
	clr.b $903 		;
	clr.b $904 		; done
	rts
				;
				; Simple (crude!) random generator for target
				;
RANDOMTARGET
	move.b RND,d0 		; get random number
	AND.B #$07,D0		;MASKAR UT DE TRE LSB
	CMP.B	#6,D0		;JÄMFÖR OM TALET ÄR MINDER ÄN 6
	BLE	RANDOMX	
	SUB	#1,D0
RANDOMX	
	CMP.B	#4,D0
	BLT	ADDTWO
	move.b d0,TPOSX 	; TPOSX now in interval
	move.b RND,d0 		; get random number
	AND.B #$03,D0		;MASKAR UT DE TVÅ LSB
	move.b d0,TPOSY
	rts
ADDTWO
	ADD	#2,D0
	JMP	RANDOMX
				;
				; Init PIA KLART
				;
PIAINIT
	CLR.B $10084 		; Nollställ CRA
	MOVE.B #$0F,$10080 	; A0-A2=MUX,A4 ANVÄNDS INTE OCH A4-A7 ÄR AD IN F7
	MOVE.B #$07,$10084 	;CRA-7 1-ställs då CA1 0->1,IRQA triggars då, piaa utpekas(nivå 5),ca2 är utgång 07

	CLR.B $10086 		; Nollställ CRB
	MOVE.B #$7F,$10082 	; B7=IN,B6-B0=UT
	MOVE.B #$3C,$10086	;INGA AVBROTT, CB2=UTGÅNG
	rts
				;
				; Install and enable ints
				;
INSTALLINTS
	MOVE.L #109,$7100	;sätter frekvensen till 500hz
	MOVE.L #200,$7104	;sätter antalet perioder till 25
	MOVE.L #0000,$7108
	MOVE.L	#MUX,$74	
	AND.W 	#$F8FF,SR
	rts

BEEP
	MOVE.L $7104,D1
BEEPSUB	BCHG #3,$10080		; INVERTERAR utbiten
	BSR DELAY 		; Vänta en halv period
	BCHG #3,$10080		; INVERTERAR utbiten
	BSR DELAY 		; Vänta en halv period
	SUBQ.L #1,D1 		; Räkna ned D1
	BNE BEEPSUB 		; Om D1>0: fortsätt pipa
	RTS

DELAY: 	MOVE.L $7100,D3 	; Hämta fördröjning T
WAIT: 	SUBQ.L #1,D3 		; Räkna ned med 1
	BNE WAIT 		; Om D3<>0: räkna vidare
	RTS 			; Hoppa tillbaka
	
	
END
