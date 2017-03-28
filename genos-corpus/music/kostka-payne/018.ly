\version "2.12.0"
\header {
  title = "Unsere Vater hofften auf dich, Op 109, Nº 1"
  composer = "Brahms"
}

<<
  \new PianoStaff { 
    <<
      \set Score.currentBarNumber = #48
      \bar ""
      \new Staff { 
        \relative c' { 
          \key f \major
          << 
            { 
              r1\f
              r4 f g a
              d1
              c1\fermata
            } 
            \\ 
            { 
              r2 r4 f,
              c'2 c
              r4 f,4 g2
              a1
            } 
          >> 
        }
      }
      \new Staff { 
        \relative c' { 
          \clef bass 
          \key f \major
          << 
            { 
              c1
              c4 r r a
              bes2 g4 g'
              c,1
            } 
            \\ 
            { 
              f,4. e8 d2
              f2 c4 c
              d2 bes
              f'1\fermata
            } 
          >> 
        }
      }
    >>
  }

  \new PianoStaff { 
    <<
      \new Staff { 
        \relative c'' { 
          \key f \major
          << 
            { 
              r1\f
              r2 r4 c4
              c2 bes2
              a1\fermata
            }
            \\ 
            { 
              r4 c, c2
              c4. d8 e4 f
              r4 d4 e2
              f1
            } 
          >> 
        }
      }
      \new Staff { 
        \relative c' { 
          \clef bass 
          \key f \major
          << 
            { 
              g2 f4 bes
              a2 g4 f
              f2 bes
              c1
            }
            \\ 
            { 
              c,1
              c2. f,4
              bes1
              f1\fermata
              \bar "||"
            } 
          >> 
        }
      }
    >>
  }
>>
