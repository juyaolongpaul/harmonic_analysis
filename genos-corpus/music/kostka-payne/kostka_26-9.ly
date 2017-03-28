\new PianoStaff {
	<<
	\new Staff{
	\key bes \major
	<<
	\relative c'{r4 fis g a bes c8 bes a4 g fis r} \\
	\relative c'{d1 cis d2 }
	>>
	}

	\new Staff {
	\clef bass \key bes \major
	<<
	\relative c' {bes2. a4 g2 fis4 g a2} \\
	\relative c {e1 ees1 d2  }
	>>
	}
	>>
}