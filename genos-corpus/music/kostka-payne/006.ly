\version "2.12.0"

\header {
  title = "Chorale"
  subtitle = "Op. 68, N°4"
  composer = "Schumann"
}

\new PianoStaff <<
  \new Staff {
    \key g \major
    \relative c'' {
      <<
        { 
          b2\mf c
          d c
          b a
          b1\fermata
          d2\< d
          e\f d
          c b
          a1\fermata
        }
        \\
        { 
          g2 fis
          g fis
          g g4( fis)
          g1
          g2 g
          g g
          fis g
          fis1
        }
      >>
    }
  }
  \new Staff  {
    \key g \major
    \clef bass
    \relative c'{
      <<
        { 
          d2 c
          b c
          d e4( d)
          d1
          b2 d
          c d
          d d
          d1
        }
        \\
        { 
          g,2 a
          b a
          g c4( d)
          g,1\fermata
          g2 b
          c b
          a g
          d1\fermata
        }
      >>
    }
  }
>>
