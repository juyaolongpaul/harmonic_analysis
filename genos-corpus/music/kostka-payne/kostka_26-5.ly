\new PianoStaff {
	<<
	\new Staff \relative c'{
	\time 3/4
	\key g \major
	<e a e'>2.^"Allegro energico e passionato"
	<fis c' fis >
	<e g b g'>
	<e a e' a >
	<cis e ais cis e ais >
	<b e b' e b'> 
	<b dis a' b > 
	<b e gis b e >
	}

	\new Staff \relative c' {
	\clef bass \key g \major
	c2.
	<a c>
	<e b>
	<c c>
	<fis, fis' ais>
	<g g' b>
	<f f' a>
	<e e' gis>
	}
	>>
}