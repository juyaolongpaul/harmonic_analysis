\version "2.10.33"

\header {
  title = "275 - O Welt, sieh hier dein Leben"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key a \major
}


soprano = \relative c'' {
  \partial 4 cis4 
  a4 b4 cis8 d8 e4 
  d2 cis4 cis8 d8 
  e4 e4 b4 cis4 
  a2 gis4 e4 
  a4 b4 cis8 d16 e16 d8 cis8   %5
  b2. cis4 
  a4 b4 cis8 d8 e4 
  d2 cis4 cis8 d8 
  e4 e4 b4 cis4 
  a2 gis4 e4   %10
  a4 b4 cis4 d4 
  cis4 b4 a2 
  
}

alto = \relative c'' {
  \partial 4 gis4 
  fis4 e4 e4 e4 
  fis4 e4 e4 a4 
  a8 gis8 a4 gis8 fis4 e8 
  e4 dis4 e4 e8 d8 
  cis4 e4 e4 a4   %5
  gis2. gis4~ 
  gis8 fis4 e8 e4 fis4 
  fis2 fis4 e4 
  e8 fis8 gis4 fis4 e4 
  fis2 e4 gis4   %10
  a4. gis8 a4 a4 
  a4. gis8 e2 
  
}

tenor = \relative c' {
  \partial 4 cis4 
  cis4 b4 a4 a4 
  a4 gis4 a4 e'4 
  e4 e4 e4 gis,4 
  fis8 gis8 a4 b4 gis4 
  cis4 b4 a4 fis'4   %5
  b,2. gis4 
  cis4 b4 a8 b8 cis4 
  cis4 b4 ais4 a4 
  b4 b4 b8 a8 gis4 
  cis4 b4 b4 b4   %10
  e4 d4 e4 d4 
  e8 fis8 b,16 cis16 d8 cis2 
  
}

baixo = \relative c {
  \partial 4 eis4 
  fis4 gis4 a4 cis,4 
  b4 e4 a,4 a'4 
  cis,4 cis8 d8 e4 cis4 
  fis2 e4 e4 
  fis4 gis4 a4 d,4   %5
  e2. eis4 
  fis4 gis4 a4 ais4 
  b2 fis4 a4 
  gis4 fis8 e8 dis4 e4 
  e4 dis4 e4 e8 d8   %10
  cis4 b4 a8 g'4 fis8 
  e8 dis8 e4 a,2 
  
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