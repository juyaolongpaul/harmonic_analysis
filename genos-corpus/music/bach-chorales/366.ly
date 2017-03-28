\version "2.10.33"

\header {
  title = "366 - O Welt, sien hier dein Leben"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key a \major
}


soprano = \relative c'' {
  \partial 4 cis8  b8 
    a4 b4 cis8 d8 e4 
  d2 cis4 cis8 d8 
  e4 e4 b4 cis8 b8 
  a2 gis4 e4 
  a4 b4 cis8 b8 cis8 d8   %5
  b2 r4 cis8 b8 
  a4 b4 cis8 d8 e4 
  d2 cis4 cis8 d8 
  e4 e4 b4 cis8 b8 
  a2 gis4 e4   %10
  a4 b4 cis8 d8 e8 d8 
  cis4 b4 a4 
  
}

alto = \relative c'' {
  \partial 4 a8  gis8 
    fis4. gis8 a4. e8 
  fis4 e4 e4 a4 
  a8 gis8 a8 b8 gis8 fis8 e4 
  e4 dis4 e4 b4 
  e8 fis8 gis4 a8 gis8 a8 b8   %5
  gis2 r4 gis4 
  fis4 e4 e8 d8 cis4 
  cis8 b16 a16 b4 a4 a'4 
  b8 cis8 b8 a8 gis8 fis8 gis4 
  gis8 fis16 e16 fis4 e4 e8 d8   %10
  cis8 d8 cis8 b8 a4 a'4 
  e8 fis8 b,8 e8 cis4 
  
}

tenor = \relative c' {
  \partial 4 e4 
    fis8 e8 d4 e4 a,4 
  a4 gis4 a4 e'4 
  e4 e4 e8 d8 cis4 
  cis4 b4 b4 gis4 
  a4 d4 e8 d8 e8 fis8   %5
  e2 r4 eis4 
  cis8 d8 cis8 b8 a4 a4 
  a8 fis'4 e8 e4 fis4 
  e4 b4 b8 a8 gis8 fis8 
  e8 cis'4 b8 b4 cis8 b8   %10
  a4 gis8 fis8 e4. fis8 
  gis8 a4 gis8 e4 
  
}

baixo = \relative c' {
  \partial 4 a4 
    d8 cis8 b4 a4 cis,4 
  b4 e4 a,4 a'8 b8 
  cis4 cis,8 d8 e4 a8 gis8 
  fis4 b,4 e4 e8 d8 
  cis8 d8 cis8 b8 a4 a'8 d,8   %5
  e2 r4 cis4 
  fis4 gis4 a8 b8 a8 gis8 
  fis4 gis4 a4 fis4 
  gis8 a8 gis8 fis8 e8 fis8 e8 dis8 
  cis4 dis4 e4 cis4   %10
  fis4 e8 d8 cis8 b8 cis8 d8 
  e8 d8 e4 a,4 
  
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