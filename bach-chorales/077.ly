\version "2.10.33"

\header {
  title = "77 - In dich hab' ich gehoffet, Herr"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key a \major
}


soprano = \relative c'' {
  \partial 4 a4 
    a4 e'4 e4 d8 cis8 
  b4 cis4 d8 cis8 b4 
  a2 r4 a4 
  b4 cis4 d4 b4 
  e4 fis4 e4 cis4   %5
  cis8 d8 e4 d4 cis8 b8 
  a8 b8 cis4 b4 cis4 
  a4 b8 cis8 d4 cis4 
  b4 cis4 d4 cis4 
  b4 a4 e'4. d8   %10
  cis4 d8 cis8 b2 
  a2.
  
}

alto = \relative c' {
  \partial 4 fis4 
    e8 fis8 gis8 a8 b4 e,4 
  dis4 e4 d8 a'8 gis4 
  e2 r4 fis4 
  fis4 e8 cis8 gis'8 fis8 e8 dis8 
  e8 fis16 gis16 a8 fis8 gis4 a4   %5
  a4 ais4 b8 a8 gis8 fis8 
  e8 fis16 gis16 a8 fis8 gis4 a8 gis8 
  fis4. gis16 ais16 b4 ais4 
  b8 a8 g4 fis4 e8 a8~ 
  a8 gis4 fis8 gis8 a8 b4~   %10
  b8 e,8 a2 gis4 
  e2.
  
}

tenor = \relative c' {
  \partial 4 cis8  d8 
    e4 d8 cis8 b4 a4 
  b4 a8 g8 a4 e'8 d8 
  cis2 r4 a4 
  a4 gis4 b8 a8 b4 
  b8 e4 dis8 e4 e4   %5
  fis4 fis4 fis4 e4 
  e8 d8 cis8 dis8 e4 e4 
  d8 cis8 d8 e8 fis4 fis8 e8 
  d4 e4 a,4 a4 
  b4 cis8 dis8 e2   %10
  e8 cis8 a8 cis8 fis8 d8 b8 e8 
  cis2. 
  
}

baixo = \relative c {
  \partial 4 fis4 
    cis'4 b8 a8 gis4 a4~ 
  a8 g8 fis8 e8 fis8 d8 e4 
  a,2 r4 d4 
  dis8 e4 eis8~ eis fis8 gis8 a8 
  gis8 e8 b'8 b,8 e4 a8 gis8   %5
  fis8 e8 d8 cis8 b4 e8 d8 
  cis8 b8 a4 e'4 a,4 
  d8 e8 d8 cis8 b4 fis'4 
  g8 fis8 e4 d4 a4 
  e'2~ e8 fis8 gis8 e8   %10
  a8 gis8 fis8 e8 d8 b8 e4 
  a,2.
  
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