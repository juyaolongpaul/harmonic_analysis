\version "2.10.33"

\header {
  title = "177 - Ach bleib bei uns, Herr Jesu Chris"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key a \major
}


soprano = \relative c'' {
  \partial 4 cis4 
    cis8 d8 e4 cis4 a4 
  b4 cis4 d4 cis4 
  b4 a4 a4. b8 
  cis4 cis4 b8 a8 b4 
  a2. cis4   %5
  b4 cis4 a4 fis4 
  gis4 a4 b4 cis4 
  a4 a4 a4. b8 
  cis4 cis4 b8 a8 b4 
  a2. 
}

alto = \relative c'' {
  \partial 4 a4 
    a4 b4 a4 a4 
  g8 fis8 e4 fis4 e4 
  e8 d8 cis4 fis2 
  e4 e4 e4. d8 
  cis2. a'4   %5
  gis4 gis4 fis8 e8 dis4 
  e4. dis8 e4 eis4 
  cis4 fis4 e4 fis8 gis8 
  a4 e4 e4. d8 
  cis2. 
}

tenor = \relative c' {
  \partial 4 e4 
    e4 e4 e4 e8 d8 
  d4 a4 a4 a4 
  gis4 a4 d2 
  cis8 b8 a4 a4 gis4 
  e2. e'4   %5
  e8 d8 cis4 cis4 b4 
  b4 a4 gis4 gis4 
  a4 d4 cis8 d8 cis8 b8 
  a4 a4 a4 gis4 
  e2. 
}

baixo = \relative c {
  \partial 4 a4 
    a'4 gis4 a4 fis4 
  g4 a4 d,4 a4 
  e'4 fis8 e8 d8 e8 fis8 gis8 
  a4 cis,8 d8 e4 e4 
  a,2. a4   %5
  e'4 eis4 fis4 b8 a8 
  gis4 fis4 e4 cis4 
  fis8 e8 fis8 gis8 a8 b8 a8 gis8 
  fis4 cis8 d8 e4 e4 
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