\version "2.10.33"

\header {
  title = "312 - O Gott, du frommer Gott"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key d \major
}


soprano = \relative c'' {
  \partial 4 a4 
    fis4 d4 a'4 a4 
  b2 r4 b4 
  e,4 e4 a4 g4 
  fis4 e4 d4 a'4 
  b4 b4 a4 g4   %5
  fis2 r4 e4 
  fis4 gis4 a4 b8 cis16 d16 
  cis4 b4 a4 a4 
  a4 a4 d4 c8 b8 
  b2 r4 b4   %10
  b4 b4 e4 d8 cis8 
  cis2 r4 a4 
  b4 a4 b4 cis4 
  d2 r4 a4 
  b8 a8 g8 fis8 e4 e4   %15
  d2. 
  
}

alto = \relative c' {
  \partial 4 d4 
    d4 d8 e8 fis4 d4 
  d2 r4 d4 
  cis4 cis4 d4 e4 
  d4 cis4 a4 fis'4 
  g4 g8 fis8 e8 d8 e4   %5
  d2 r4 cis4 
  d4 e4 e4 fis4 
  e4. d8 cis4 cis4 
  d4 d4 d4 d4 
  d2 r4 g4   %10
  g8 fis8 g8 a8 b8 a8 b8 gis8 
  a2 r4 d,4 
  g4 fis4 fis4 fis4 
  fis2 r4 fis8 e8 
  d4 d4 cis4 cis4   %15
  a2. 
  
}

tenor = \relative c {
  \partial 4 fis4 
    a4 a4 d4 a4 
  g2 r4 b4 
  a4 a4 a4 a4 
  a4. g8 fis4 d'4 
  d4 g,4 a4 a4   %5
  a2 r4 a4 
  a4 b4 a4 a4 
  a4 gis4 e4 e4 
  fis8 e8 fis8 g8 a8 g8 a8 fis8 
  g2 r4 d'4   %10
  e4 e4 b4 e4 
  e2 r4 d4 
  d4 d4 d4 cis4 
  b2 r4 d8 cis8 
  b4 b8 d8 a4. g8   %15
  fis2. 
  
}

baixo = \relative c {
  \partial 4 d4 
    d8 e8 fis8 g8 fis8 e8 fis8 d8 
  g2 r4 g4 
  a8 g8 fis8 e8 fis8 e8 d8 cis8 
  d4 a4 d4 d4 
  g8 fis8 e8 d8 cis8 b8 cis8 a8   %5
  d2 r4 a4 
  d8 cis8 d8 b8 cis4 d4 
  e8 d8 e4 a,4 a4 
  d8 cis8 d8 e8 fis8 e8 fis8 d8 
  g2 r4 g8 fis8   %10
  e8 dis8 e8 fis8 gis8 fis8 gis8 e8 
  a2 r4 fis4 
  g8 b8 d8 e8 d8 cis8 b8 ais8 
  b,2 r4 fis'4 
  g8 a8 b8 g8 a4 a,4   %15
  d2. 
  
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