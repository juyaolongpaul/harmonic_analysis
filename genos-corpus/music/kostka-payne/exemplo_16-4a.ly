\version "2.10.33"
sop = \relative c'' { \clef treble
	r8 g16 c e g, c e
	r8 g,16 c e g, c e
	r8 a,16 d f a, d f
	r8 a,16 d f a, d f
	r8 g,16 d' f g, d' f
	r8 g,16 d' f g, d' f
	r8 g,16 c e g, c e
	r8 g,16 c e g, c e
}

cont = \relative c' { \clef treble
	r16 e8. ~ e4	r16 e8. ~ e4
	r16 d8. ~ d4	r16 d8. ~ d4
	r16 d8. ~ d4	r16 d8. ~ d4
	r16 e8. ~ e4	r16 e8. ~ e4
}


bass = \relative c' { \clef bass
	c2 c c c b b c c
}

\new PianoStaff { <<
	\new Staff { << { \sop } \\ { \cont } >> }
	\new Staff { \bass }
	>> }