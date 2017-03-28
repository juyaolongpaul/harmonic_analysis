\version "2.10.33"

\header {
  title = "352 - Es woll uns Gott genädig sein"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key a \minor
}


soprano = \relative c'' {
  \repeat volta 2 {
    \partial 4 b4 
    c4 b4 a4 b8 c8 
    d4 e4 d4 c4 
    b2 r4 d4 
    c4 b4 c8 b8 a4 
    g4 f4 e4 } g4   %5
  f4 e4 d4 e8 d8 
  c8 d8 d4 e4 b'4 
  c4 b8 a8 g4 a8 b8 
  c4 d4 b4 e4 
  d8 c8 b8 c8 d4 a4   %10
  g4 f4 e4 d4 
  g4 a4 b8 c8 d4 
  c4 b4 a4 c4 
  b4 a4 g8 f8 e8 f8 
  g4 f4 e2~   %15
  e2. 
  
}

alto = \relative c'' {
  \repeat volta 2 {
    \partial 4 gis4 
    a8 e8 f8 e4 d8 g4~ 
    g8 f8 g8 e8 f4 e8 dis8 
    e2 r4 f4 
    e8 fis8 g4 g4 f4 
    e4. d16 c16 b4 } e8 d8   %5
  c8 d4 c8 a8 g4 gis8 
  a4. b8 c8 b8 e4 
  e8 a8 g8 f8 e4 d4 
  c8 e8 a4 a8 gis8 g8 fis8 
  g4 g4 g4. f8   %10
  e4. d8 d8 cis8 a8 b8 
  c4 c4 b8 e8 d8 e8 
  e8 a4 gis8 a4 g8 a8 
  b8 g8 e8 f8 b,8 d8 g8 f8 
  f8 e8 e8 d8 d8 e16 d16 c4   %15
  b2. 
  
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 e4 
    e8 c8 d8 e8 a,4 g4 
    a4 bes4 a8 gis8 a4 
    a4 gis4 r4 a8 b8 
    c4 d4 e8 d8 c4 
    bes8 a4 b8 gis4 } b4   %5
  a8 g4 a8 f4 e4 
  e4 a4 a8 gis8 gis4 
  a16 b16 c4 d8 e8 e,8 fis8 gis8 
  a4. f'8 e4 g,8 a8 
  b8 c8 d4 d4 d4   %10
  e8 a,8 a4 a4 f4 
  e4. fis8 g4 a8 b8 
  c8 e8 f8 e8 c4 e4 
  d4 cis8 d8 e8 b8 c4 
  d8 a8 a8 b4 e,8 a4   %15
  a4 gis8 fis8 gis4 
  
}

baixo = \relative c {
  \repeat volta 2 {
    \partial 4 e4 
    a4. g8 f4. e8 
    d4. cis8 d4 a4 
    e'2 r4 d4 
    a'4 g8 f8 e4 f8 e8 
    d8 cis8 d4 e4 } e4   %5
  a,8 b8 c4. b8 c8 b8 
  a8 g8 f4 e4 e'4 
  a,4. b8 c4. b8 
  a8 a'8 f8 d8 e4 c4 
  g'4 g,8 a8 b8 c8 d4~   %10
  d8 cis8 d4 a4 d4 
  c8 b8 a4 e'4 fis8 gis8 
  a8 c,8 d8 e8 f4 e8 fis8 
  g4. f8 e8 d8 c8 a8 
  b8 cis8 d8 gis,8 a8 b8 c8 d8   %15
  e2. 
  
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