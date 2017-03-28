\version "2.10.33"

\header {
  title = "278 - Wie schön leuchtet der Morgenstern"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key e \major
}


soprano = \relative c' {
  \repeat volta 2 {
    \partial 4 e4 
    b'4 gis4 e4 b'4 
    cis4 cis4 b4 b4 
    cis4 dis4 e4 dis4 
    cis4 cis4 b4 gis4 
    cis4 b4 a4 gis4     %5
    fis2 e4 } r4 
  b'2 gis2 
  b2 gis2 
  a4 gis4 fis4 gis4 
  a4 gis4 fis4 gis4   %10
  a4 gis4 fis2 
  e2 e'4 dis4 
  cis4 b4 a4 gis4 
  fis2 e4 

}

alto = \relative c' {
  \repeat volta 2 {
    \partial 4 b4 
    b4 b4 cis4 dis4 
    gis4 fis4 fis4 gis4 
    a4 a4 gis4 fis4 
    fis8 gis8 fis8 e8 dis4 e4 
    e4 dis4 e8 fis8 gis8 e8     %5
    e4 dis4 b4 } r4 
  fis'2 e2 
  fis2 e2 
  fis4 e4 dis4 e4 
  fis4 e4 dis4 e4   %10
  fis4 e8 dis8 cis8 dis16 e16 dis4 
  b2 gis'4 fisis 
  gis gis8 fis8 e4 e 
  e dis b 

}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 gis4 
    fis4 e8 fis8 gis8 ais8 b4 
    b4 ais4 dis4 e4 
    e4 fis4 b,4 b4 
    b4 ais4 b4 b4 
    a8 gis8 fis4 cis'4 b4     %5
    b4. a8 gis4 } r4 
  b2 b2 
  dis2 cis2 
  cis4 b4 b4 b4 
  b4 b4 b4 b4   %10
  cis8 dis8 e4. cis8 b8 a8 
  gis2 cis8 b8 ais4 
  gis4 dis'4 cis4 b4 
  cis4 b8 a8 gis4 

}

baixo = \relative c {
  \repeat volta 2 {
    \partial 4 e4 
    dis4 e4 cis4 gis'4 
    e4 fis4 b,4 e4 
    a4 gis8 fis8 gis8 ais8 b4 
    fis4 fis,4 b4 e4 
    a,4 b4 cis8 dis8 e4     %5
    b2 e4 } r4 
  dis2 e2 
  b2 cis2 
  fis4 gis4 a8 gis8 fis8 e8 
  dis4 e4 b8 b'8 a8 gis8   %10
  fis4 cis8 b8 a4 b4 
  e2 cis4 dis4 
  eis8 fisis8 gis4 cis,8 dis8 e4 
  ais,4 b4 e4 

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