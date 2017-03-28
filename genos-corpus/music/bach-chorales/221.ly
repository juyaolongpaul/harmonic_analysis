\version "2.10.33"

\header {
  title = "221 - Herr, straf mich nicht in deinem Zorn"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key a \minor
}


soprano = \relative c'' {
  \repeat volta 2 {
    \partial 4 a4 
    c4 b4 a4 a4 
    g4 f4 e4 e4 
    fis4 gis4 a4 c4 
    b2 a4 } b4 
  b4 b4 b4 g4   %5
  a4 b4 c4 b4 
  b4 b4 c4 b4 
  a4 a4 gis4 gis4 
  a4 b4 c4 d4 
  b2 a2   %10
  
}

alto = \relative c' {
  \repeat volta 2 {
    \partial 4 e4 
    a4 a8 g8 f4 f4~ 
    f8 e4 d8 cis4 e4 
    d8. e16 f8 e8 e8 fis16 gis16 a4 
    a4 gis4 e4 } g4 
  fis4 g4 fis4. e8~   %5
  e8 d4 e8 e4 e4 
  fis16 g16 a4 gis8 a4. g8~ 
  g8 f16 e16 d4 e4 e4 
  e4 e4. a16 gis16 a4~ 
  a8 gis16 fis16 gis4 e2   %10
  
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 c4 
    e4 e8. d16 c4 d4~ 
    d8 a8 a4 a4 a4 
    a4 d8 c16 b16 c8 d8 e4 
    f8 d8 b8 e16 d16 c4 } e4 
  dis4 e4 b4 b8 c8~   %5
  c8 d16 c16 b16 a16 b8 a4 g4 
  b4 e4 e4 e8. d16 
  c4 b8 a8 b4 b4 
  c4 b4 a4. f'8 
  e8 b8 e8 d8 cis2   %10
  
}

baixo = \relative c {
  \repeat volta 2 {
    \partial 4 a4 
    a'4 e4 f8 e8 d8 c8 
    b8 cis8 d4 a4 c4~ 
    c8 d16 c16 b8 e8 c8 b8 a8 c8 
    d8 b8 e4 a,4 } e'4 
  b'8 a8 g8 fis8 e8 dis8 e4   %5
  fis4 gis4 a4 e4 
  dis4 d4 c8 d8 e4 
  f8 e8 f4 e4 e8 d8 
  c8 b8 a8 gis8 a4 f'8 d8 
  e2 a,2   %10
  
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