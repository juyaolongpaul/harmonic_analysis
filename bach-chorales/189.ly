\version "2.10.33"

\header {
  title = "189 - Herr Jesu Christ, wahr'r Mensch und Gott"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key a \major
}


soprano = \relative c'' {
  \partial 4 a4 
    a4 a4 fis4 b4 
  a4 a4 gis4 cis4 
  cis4 cis4 fis4 e4 
  e4 dis4 e4 cis4 
  cis4 cis4 d4 cis4   %5
  b4 ais4 b4 cis4 
  cis4 cis4 b4 a4 
  a4 gis4 a
  
}

alto = \relative c' {
  \partial 4 e4 
    e8 d8 cis8 b8 a4 fis'8 gis8~ 
  gis8 fis16 e16 fis4 e4 e4 
  a4 a4 a4 gis4 
  gis4 fis8 a8 a8 gis8 gis4 
  fis4 e4 fis4 g4   %5
  fis8 g8 fis8 e4 d8 a'4 
  a4 a4. gis4 fis8 
  e8 fis4 e8 e4
  
}

tenor = \relative c' {
  \partial 4 cis8  d8 
    e4 e4 d4 dis4 
  e4 b4 b4 a4 
  e'4 fis4 b,4 b4 
  b4 b4 b4 cis8 b8 
  a8 gis8 ais4. b4 ais8   %5
  b8 e8 cis4 b4 e4 
  e4 fis4 fis8 cis8 cis4 
  cis8 b8 b8 cis16 d16 cis4
  
}

baixo = \relative c {
  \partial 4 a8  b8 
    cis8 b8 a4 d8 cis8 b4 
  cis4 dis4 e4 a4~ 
  a8 gis8 fis8 e8 dis4 e8 fis8 
  gis8 a8 b4 e,4 f4 
  fis4 gis4 fis4 e4   %5
  d8 e8 fis4 b,4 a4 
  a'8 gis8 fis8 e8 dis8 eis8 fis4 
  cis8 d8 e4 a,
  
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