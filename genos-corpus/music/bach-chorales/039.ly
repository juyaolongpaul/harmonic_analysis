\version "2.10.33"

\header {
  title = "39 - Ach was soll ich Sünder machen"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key e \minor
}


soprano = \relative c' {
  e4 e4 g4 g4 
  a4 a4 b4 b4 
  b4 d4 c4 b4 
  a4. a8 g2 
  a4 a4 b4 g4   %5
  fis4. fis8 e2 
  e'4 e4 d4 d4 
  cis4 cis4 b4 b4 
  d4 d4 c4 b4 
  a4. a8 g2   %10
  a4 a4 b4 g4 
  fis4. fis8 e2 
  
}

alto = \relative c' {
  b4 b4 e4 e4 
  e4 a8 fis8 g4 g4 
  g4 g4 g8 fis8 g4 
  g4 fis4 d2 
  fis4 e4 dis8 fis8 b,8 e16 dis16   %5
  e4 dis4 b2 
  g'4 g4 fis4 fis4 
  g4 fis4 fis4 fis4 
  fis4 g4 g8 a4 g8 
  g4 fis4 d2   %10
  e4 fis4 fis4. e8 
  e4 dis4 b2 
  
}

tenor = \relative c' {
  g4 g4 b4 b4 
  c4 d4 d4 d4 
  d4 d8 b8 c8 d8 d4 
  e8 c8 a8 d16 c16 b2 
  d4 a4. g16 fis16 g8 c8~   %5
  c8 a8 fis8 b16 a16 g2 
  b4 b4 b4 b4 
  b4. ais8 d4 d4 
  d8 c8 b4 e8 d4 e8~ 
  e8 c8 a8 d16 c16 b2   %10
  cis4 d8 c8 b4 b8 c8~ 
  c8 a8 fis8 b16 a16 gis2 
  
}

baixo = \relative c {
  e8 fis8 g8 fis8 e8 d8 c8 b8 
  a8 g8 fis8 d8 g4 g'4 
  g,8 a8 b8 e8 a,8 d8 g,8 b8 
  c8 a8 d4 g,2 
  fis'8 e8 d8 c8 b8 dis8 e8 c8   %5
  a8 fis8 b4 e,2 
  e8 fis8 g8 a8 b8 cis8 d8 b8 
  e8 cis8 fis4 b,2 
  b'8 a8 g8 fis8 e8 fis8 g8 e8 
  c8 a8 d4 g,2   %10
  g'4 fis8 e8 dis4 e8 c8 
  a8 fis8 b4 e,2 
  
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