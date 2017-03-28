\version "2.10.33"

\header {
  title = "238 - Es wird schier der letzte Tag"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key e \minor
}


soprano = \relative c' {
  e4 e4 b'4 b8 c8 
  d4 b4 g4 a8 b8 
  c2 b2 
  a4 a4 g4 a4 
  b4 a4 g4 fis4   %5
  e2 d2 
  g4 fis4 g8 a8 b4 
  a8 g8 fis4 e2 
  fis4 e8 fis8 g4 fis4 
  e1   %10
  
}

alto = \relative c' {
  b4 b4 e8 fis8 g4 
  a8 fis8 g8 fis8 e4 fis8 g8~ 
  g8 fis16 e16 fis4 g2 
  g4. fis8 fis8 e8 d8 a'8 
  a8 g4 fis8 e4 a,8 d8   %5
  d8 cis16 b16 cis4 a2 
  e'4 fis4 e4 e4 
  e4 dis4 e2 
  dis4 e8 dis8 e4. dis8 
  b1   %10
  
}

tenor = \relative c' {
  g4 g8 a8 b4 e4 
  d4 d4 c4. d8 
  c8 d16 e16 d4 d2 
  e4 d8. c16 b4 d4 
  d4 d4. cis8 d8 a8   %5
  b8 g8 e8 a16 g16 fis2 
  b4 b4 b4 b4 
  c4 c8 b16 a16 g2 
  b4. a8 b4~ b8. a16 
  gis1   %10
  
}

baixo = \relative c {
  e4. fis8 g4 e4 
  fis8 d8 g4 c8 b8 a8 g8 
  a4 d,4 g2 
  cis,4 d4 e4 fis4 
  g4 d4 e4 fis4   %5
  g8 e8 a8 a,8 d2 
  e4. dis8 e4 g,4 
  a4 b4 c2 
  b8 a8 g8 fis8 e4 b'4 
  e1   %10
  
}

\score {
  <<
    \new StaffGroup <<
      \override StaffGroup.SystemStartBracket #'style = #'line 
      \new Staff {
        <<
          \global
          \new Voice = "soprano" { \voiceOne \soprano }
          \new Voice = "alto" { \voiceTwo \alto }
        >>
      }
      \new Staff {
        <<
          \global
          \clef "bass"
          \new Voice = "tenor" {\voiceOne \tenor }
          \new Voice = "baixo" { \voiceTwo \baixo \bar "|."}
        >>
      }
    >>
  >>
  \layout {}
  \midi {}
}